from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fast_student_dg import (
    ATTRIBUTE_NAMES,
    DOMAIN_NAMES,
    ImageResolver,
    UPARFastDataset,
    EfficientNetB0DG,
    MaskedAsymmetricLoss,
    ModelEMA,
    TrainConfig,
    build_train_transform,
    build_eval_transform,
    make_domain_balanced_sampler,
    domain_loss,
    seed_everything,
)
from vapor_par.metrics import (
    evaluate_probs,
    tune_attribute_thresholds,
    robust_selection_score,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default="/content/UPAR-Challenge-2027")
    p.add_argument("--checkpoint-dir", default="/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--domain-loss-weight", type=float, default=0.08)
    p.add_argument("--grl-lambda", type=float, default=0.35)
    p.add_argument("--init-checkpoint", default="")
    return p.parse_args()


def load_init_weights(model, path: Path):
    if not path.exists():
        return False
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("model_state") or obj.get("model") or obj.get("state_dict") or obj.get("model_state_dict")
    if not isinstance(state, dict):
        return False

    # Old fast student = torchvision EfficientNet state_dict. Convert to DG wrapper.
    if "features.0.0.weight" in state and "classifier.1.weight" in state:
        converted = {}
        for k, v in state.items():
            if k.startswith("features."):
                converted[k] = v
            elif k.startswith("classifier.1."):
                converted[k.replace("classifier.1", "attr_head")] = v
        missing, unexpected = model.load_state_dict(converted, strict=False)
        # Domain head / MixStyle have no pretrained state requirement.
        critical_missing = [k for k in missing if k.startswith("features.") or k.startswith("attr_head.")]
        if critical_missing:
            raise RuntimeError(f"Could not initialize critical weights: {critical_missing[:20]}")
        print(f"Initialized DG student from old Fast Student: {path}")
        return True

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Initialized model from {path}; missing={len(missing)} unexpected={len(unexpected)}")
    return True


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets, all_domains = [], [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        targets = batch["target"].numpy()
        valid = batch["valid"].numpy()
        targets = np.where(valid > 0, targets, -1).astype(np.int32)
        all_probs.append(probs)
        all_targets.append(targets)
        all_domains.append(batch["domain"].numpy())
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_domains, axis=0),
    )


def metrics_report(targets, probs, domains, thresholds=0.5):
    overall = evaluate_probs(targets, probs, thresholds)
    per_domain = {}
    for d in sorted(set(int(x) for x in domains.tolist() if int(x) >= 0)):
        mask = domains == d
        per_domain[DOMAIN_NAMES[d]] = evaluate_probs(targets[mask], probs[mask], thresholds)
    robust, _, _ = robust_selection_score(targets, probs, thresholds, domains=domains)
    return overall, per_domain, robust


def main():
    args = parse_args()
    seed_everything(args.seed)

    repo_root = Path(args.repo_root)
    data_root = repo_root / "data"
    train_csv = data_root / "annotations" / "task1" / "train" / "gt.csv"
    val_csv = data_root / "annotations" / "task1" / "val" / "gt.csv"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Official train/val gt.csv not found. Run prepare_data.py first.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    missing = [c for c in ATTRIBUTE_NAMES if c not in train_df.columns or c not in val_df.columns]
    if missing:
        raise RuntimeError(f"Missing attributes: {missing}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        domain_loss_weight=args.domain_loss_weight,
        grl_lambda=args.grl_lambda,
        num_workers=args.num_workers,
    )

    resolver = ImageResolver(data_root, repo_root)
    train_ds = UPARFastDataset(
        train_df,
        resolver,
        build_train_transform(cfg.image_height, cfg.image_width),
    )
    val_ds = UPARFastDataset(
        val_df,
        resolver,
        build_eval_transform(cfg.image_height, cfg.image_width),
    )

    print("Train domain counts:")
    for d, name in enumerate(DOMAIN_NAMES):
        print(f"  {name}: {(train_ds.domains == d).sum():,}")
    print("Val domain counts:")
    for d, name in enumerate(DOMAIN_NAMES):
        print(f"  {name}: {(val_ds.domains == d).sum():,}")

    sampler = make_domain_balanced_sampler(train_ds.domains, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB0DG(pretrained=True)

    init_path = Path(args.init_checkpoint) if args.init_checkpoint else (
        ckpt_dir / "fast_student_best.pt"
    )
    if init_path.exists():
        load_init_weights(model, init_path)

    model = model.to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    ema.module = ema.module.to(device)

    criterion = MaskedAsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    prefix = f"fast_student_dg_seed{args.seed}"
    last_path = ckpt_dir / f"{prefix}_last.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    start_epoch = 0
    best_selection = -1.0

    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        ema.module.load_state_dict(state["ema_model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_selection = float(state.get("best_selection", -1.0))
        print(f"Resumed {last_path} from epoch {start_epoch}; best_selection={best_selection:.6f}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"DG seed{args.seed} {epoch+1}/{cfg.epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            domains = batch["domain"].to(device, non_blocking=True)

            # Smooth GRL ramp avoids destabilizing early pretrained features.
            epoch_progress = (epoch + steps / max(len(train_loader), 1)) / max(cfg.epochs, 1)
            grl = cfg.grl_lambda * (2.0 / (1.0 + math.exp(-10.0 * epoch_progress)) - 1.0)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                attr_logits, dom_logits, _ = model(
                    images,
                    grl_lambda=grl,
                    return_domain=True,
                )
                attr_loss = criterion(attr_logits, targets, valid)
                d_loss = domain_loss(dom_logits, domains)
                loss = attr_loss + cfg.domain_loss_weight * d_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running += float(loss.item())
            steps += 1
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                attr=f"{attr_loss.item():.4f}",
                domain=f"{d_loss.item():.4f}",
                grl=f"{grl:.3f}",
            )

        scheduler.step()

        probs, targets_np, domains_np = evaluate(ema.module, val_loader, device)
        overall, per_domain, robust = metrics_report(
            targets_np, probs, domains_np, thresholds=0.5
        )
        # Model selection is intentionally domain-robust, not optimistic mixed-source avg.
        selection = 0.5 * overall["Challenge_Avg"] + 0.5 * robust

        print({
            "epoch": epoch + 1,
            "train_loss": running / max(steps, 1),
            "official_mA": overall["mA"],
            "official_Instance_F1": overall["instance_f1"],
            "official_Challenge_Avg": overall["Challenge_Avg"],
            "robust_source_score": robust,
            "selection_score": selection,
            "per_domain": {k: v["Challenge_Avg"] for k, v in per_domain.items()},
        })

        payload = {
            "format": "upar-fast-student-dg-v1",
            "epoch": int(epoch),
            "seed": int(args.seed),
            "model_state": model.state_dict(),
            "ema_model_state": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_selection": float(max(best_selection, selection)),
            "official_metrics_05": overall,
            "robust_source_score_05": float(robust),
            "per_domain_05": per_domain,
            "attribute_names": ATTRIBUTE_NAMES,
            "image_height": cfg.image_height,
            "image_width": cfg.image_width,
            "config": vars(cfg),
        }
        torch.save(payload, last_path)

        if selection > best_selection:
            best_selection = float(selection)
            torch.save(payload, best_path)
            print(f"Saved new DG best: {best_path}")

    # Final robust threshold calibration from the best EMA model.
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["ema_model_state"], strict=True)
    model = model.to(device).eval()
    probs, targets_np, domains_np = evaluate(model, val_loader, device)
    thresholds = tune_attribute_thresholds(
        targets_np,
        probs,
        domains=domains_np,
    )
    overall, per_domain, robust = metrics_report(
        targets_np, probs, domains_np, thresholds=thresholds
    )

    best["inference_model_state"] = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }
    best["thresholds"] = torch.tensor(thresholds, dtype=torch.float32)
    best["official_metrics_calibrated"] = overall
    best["per_domain_calibrated"] = per_domain
    best["robust_source_score_calibrated"] = float(robust)
    torch.save(best, best_path)

    print("\nFINAL DG CHECKPOINT")
    print("Path:", best_path)
    print("Overall:", json.dumps(overall, indent=2))
    print("Per-domain Challenge_Avg:", {k: v["Challenge_Avg"] for k, v in per_domain.items()})
    print("Robust source score:", robust)
    print("Threshold range:", float(thresholds.min()), float(thresholds.max()))


if __name__ == "__main__":
    main()
