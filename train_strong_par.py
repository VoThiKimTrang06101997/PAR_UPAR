from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from strong_par import (
    ATTRIBUTE_NAMES,
    DOMAIN_NAMES,
    ImageResolver,
    MaskedBalancedBCE,
    ModelEMA,
    StrongPARModel,
    TrainConfig,
    UPARDataset,
    average_state_dicts,
    build_eval_transform,
    build_train_transform,
    dataset_pos_weight,
    load_legacy_teacher,
    make_tempered_domain_sampler,
    response_kd_loss,
    seed_everything,
)
from vapor_par.metrics import (
    evaluate_probs,
    robust_selection_score,
    tune_lodo_thresholds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["convnext_tiny", "swin_v2_t"], required=True)
    p.add_argument("--repo-root", type=Path, default=Path("/content/UPAR-Challenge-2027"))
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sampler-alpha", type=float, default=0.35)
    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    p.add_argument("--no-kd", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def paths_for(args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"strong_{args.backbone}_seed{args.seed}"
    return (
        args.checkpoint_dir / f"{stem}_last.pt",
        args.checkpoint_dir / f"{stem}_best.pt",
        stem,
    )


@torch.inference_mode()
def evaluate(model, loader, device, tta_flip=False):
    model.eval()
    probs, ys, ds = [], [], []
    for batch in tqdm(
        loader,
        desc="Validation" + (" + flip-TTA" if tta_flip else ""),
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.25,
    ):
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            z = model(x)
            if tta_flip:
                zf = model(torch.flip(x, dims=[3]))
                z = 0.5 * (z + zf)
        probs.append(torch.sigmoid(z).float().cpu().numpy())

        y = batch["target"].numpy()
        v = batch["valid"].numpy()
        y = np.where(v > 0, y, -1)
        ys.append(y.astype(np.int32))
        ds.append(np.asarray(batch["domain"], dtype=np.int64))

    return np.concatenate(probs), np.concatenate(ys), np.concatenate(ds)


def make_loaders(args, cfg):
    data_root = args.repo_root / "data"
    train_csv = data_root / "annotations" / "task1" / "train" / "gt.csv"
    val_csv = data_root / "annotations" / "task1" / "val" / "gt.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "Official task1 train/val annotations not found. Run prepare_data.py first."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    resolver = ImageResolver(data_root, args.repo_root)
    train_ds = UPARDataset(
        train_df,
        resolver,
        build_train_transform(cfg.image_height, cfg.image_width),
    )
    val_ds = UPARDataset(
        val_df,
        resolver,
        build_eval_transform(cfg.image_height, cfg.image_width),
    )

    sampler = make_tempered_domain_sampler(
        train_ds.domains,
        seed=args.seed,
        alpha=float(args.sampler_alpha),
    )

    bs = args.batch_size or cfg.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(bs, 64),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    counts = {
        DOMAIN_NAMES[d]: int((train_ds.domains == d).sum())
        for d in range(len(DOMAIN_NAMES))
    }
    print("Train:", train_df.shape, "Val:", val_df.shape, flush=True)
    print("Train domain counts:", counts, flush=True)
    print("Tempered sampler alpha:", args.sampler_alpha, flush=True)

    pos_weight = dataset_pos_weight(train_ds.targets)
    print(
        "pos_weight range:",
        float(pos_weight.min()),
        float(pos_weight.max()),
        flush=True,
    )

    return train_loader, val_loader, pos_weight


def _candidate_score(path: Path):
    m = re.search(r"_score([0-9.]+)\.pt$", path.name)
    return float(m.group(1)) if m else -1e9


def candidate_paths(checkpoint_dir: Path, stem: str):
    return sorted(
        checkpoint_dir.glob(f"{stem}_ep*_score*.pt"),
        key=_candidate_score,
        reverse=True,
    )


def prune_candidates(checkpoint_dir: Path, stem: str, keep: int):
    paths = candidate_paths(checkpoint_dir, stem)
    for p in paths[keep:]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return paths[:keep]


def save_candidate(path, model_state, epoch, selection, metrics_05):
    payload = {
        "epoch": int(epoch),
        "selection": float(selection),
        "ema_model_state": {k: v.detach().cpu() for k, v in model_state.items()},
        "metrics_05": metrics_05,
    }
    torch.save(payload, path)


def calibrate_and_save(
    *,
    backbone,
    seed,
    state,
    model,
    val_loader,
    device,
    best_path,
    selection,
    epoch,
    cfg,
    source,
):
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Calibration must match deployment: horizontal-flip TTA is included here.
    probs, yt, domains = evaluate(model, val_loader, device, tta_flip=True)
    thresholds, report = tune_lodo_thresholds(yt, probs, domains)
    calibrated = evaluate_probs(yt, probs, thresholds)
    robust, _, per_domain = robust_selection_score(
        yt, probs, thresholds, domains
    )

    package = {
        "format": "upar-strong-v2",
        "backbone": backbone,
        "seed": int(seed),
        "epoch": int(epoch),
        "selection": float(selection),
        "selection_calibrated_robust": float(robust),
        "source": source,
        "inference_model_state": {
            k: v.detach().cpu() for k, v in state.items()
        },
        "thresholds": torch.tensor(thresholds, dtype=torch.float32),
        "metrics_calibrated": calibrated,
        "lodo_calibration": report,
        "per_domain_calibrated": per_domain,
        "attribute_names": ATTRIBUTE_NAMES,
        "image_height": cfg.image_height,
        "image_width": cfg.image_width,
    }
    torch.save(package, best_path)

    print("\nCALIBRATED BEST:", best_path, flush=True)
    print(json.dumps(calibrated, indent=2), flush=True)
    print(
        "LODO:",
        {k: report.get(k) for k in ("global_t", "alpha", "shift", "objective")},
        flush=True,
    )
    print(
        "Threshold range:",
        float(np.min(thresholds)),
        float(np.max(thresholds)),
        flush=True,
    )
    return package


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = TrainConfig(num_workers=args.num_workers)
    if args.epochs > 0:
        cfg.epochs = int(args.epochs)
    if args.batch_size > 0:
        cfg.batch_size = int(args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_path, best_path, stem = paths_for(args)

    print("=" * 96, flush=True)
    print(
        f"UPAR STRONG v2 | backbone={args.backbone} | seed={args.seed} "
        f"| device={device}",
        flush=True,
    )
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("[1/5] Loading data ...", flush=True)
    train_loader, val_loader, pos_weight = make_loaders(args, cfg)

    print("[2/5] Building model ...", flush=True)
    model = StrongPARModel(args.backbone, pretrained=True).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    ema.module = ema.module.to(device)

    teacher = None
    if not args.no_kd and args.teacher_checkpoint is not None:
        if args.teacher_checkpoint.exists():
            teacher = load_legacy_teacher(args.teacher_checkpoint, device)
            print(
                "Low-weight KD teacher loaded:",
                args.teacher_checkpoint,
                flush=True,
            )
        else:
            print(
                "Teacher checkpoint not found; continuing without KD:",
                args.teacher_checkpoint,
                flush=True,
            )

    criterion = MaskedBalancedBCE(
        pos_weight=pos_weight,
        label_smoothing=0.03,
    ).to(device)

    lr = cfg.lr_convnext if args.backbone == "convnext_tiny" else cfg.lr_swin
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=lr * 0.04,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    start_epoch = 0
    best_selection = -1e9

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        ema.module.load_state_dict(ck["ema_model_state"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1
        best_selection = float(ck.get("best_selection", -1e9))
        print(
            f"Resuming {last_path} from epoch {start_epoch+1}/{cfg.epochs}",
            flush=True,
        )

    print("[3/5] Training ...", flush=True)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_t0 = time.time()
        run_loss = run_sup = run_kd = 0.0
        steps = 0

        prog = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"{args.backbone} seed{args.seed} | epoch {epoch+1}/{cfg.epochs}",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.25,
            leave=True,
        )

        for batch in prog:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(x)
                sup = criterion(logits, y, valid)

                kd = logits.sum() * 0.0
                kd_weight = 0.0
                if (
                    teacher is not None
                    and epoch >= cfg.kd_warmup_epochs
                ):
                    with torch.no_grad():
                        tlogits = teacher(x)
                    kd = response_kd_loss(
                        logits,
                        tlogits,
                        valid,
                        temperature=cfg.kd_temperature,
                    )
                    # Ramp gently; never let KD dominate supervised BCE.
                    ramp = min(
                        1.0,
                        (epoch - cfg.kd_warmup_epochs + 1) / 3.0,
                    )
                    kd_weight = cfg.kd_weight * ramp

                loss = sup + kd_weight * kd

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch+1}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            run_loss += float(loss.detach())
            run_sup += float(sup.detach())
            run_kd += float(kd.detach())
            steps += 1

            prog.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                sup=f"{float(sup.detach()):.4f}",
                kd=f"{float(kd.detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )

        prog.close()
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{cfg.epochs} train finished "
            f"in {(time.time()-epoch_t0)/60:.1f} min",
            flush=True,
        )
        print("Running source-domain validation ...", flush=True)

        probs, yt, domains = evaluate(
            ema.module,
            val_loader,
            device,
            tta_flip=False,
        )
        m05 = evaluate_probs(yt, probs, 0.5)
        robust, _, per_domain = robust_selection_score(
            yt, probs, 0.5, domains
        )
        selection = (
            0.50 * m05["Challenge_Avg"]
            + 0.50 * robust
        )

        summary = {
            "backbone": args.backbone,
            "epoch": epoch + 1,
            "seed": args.seed,
            "selection": selection,
            "metrics_05": m05,
            "robust_05": robust,
            "per_domain_05": {
                DOMAIN_NAMES[d]: m["Challenge_Avg"]
                for d, m in per_domain.items()
            },
            "train": {
                "loss": run_loss / max(steps, 1),
                "sup": run_sup / max(steps, 1),
                "kd": run_kd / max(steps, 1),
            },
        }
        print(json.dumps(summary, indent=2), flush=True)

        last_payload = {
            "format": "upar-strong-v2-last",
            "backbone": args.backbone,
            "epoch": epoch,
            "seed": args.seed,
            "best_selection": max(best_selection, selection),
            "model_state": model.state_dict(),
            "ema_model_state": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics_05": m05,
            "robust_05": robust,
            "attribute_names": ATTRIBUTE_NAMES,
            "image_height": cfg.image_height,
            "image_width": cfg.image_width,
        }
        torch.save(last_payload, last_path)

        cand = (
            args.checkpoint_dir
            / f"{stem}_ep{epoch+1:02d}_score{selection:.6f}.pt"
        )
        save_candidate(
            cand,
            ema.module.state_dict(),
            epoch,
            selection,
            m05,
        )
        top = prune_candidates(
            args.checkpoint_dir,
            stem,
            cfg.topk_soup,
        )

        if selection > best_selection:
            best_selection = float(selection)
            print("New best source selection:", best_selection, flush=True)
            # Crucial fix: best.pt is calibrated immediately, not only after the
            # final epoch. A Colab interruption therefore cannot leave an
            # uncalibrated 0.5-threshold checkpoint like the previous run.
            calibrate_and_save(
                backbone=args.backbone,
                seed=args.seed,
                state=ema.module.state_dict(),
                model=model,
                val_loader=val_loader,
                device=device,
                best_path=best_path,
                selection=selection,
                epoch=epoch,
                cfg=cfg,
                source="single_best",
            )

        print("Top soup candidates:", [p.name for p in top], flush=True)

    print("[4/5] Building top-k weight soup ...", flush=True)
    top = prune_candidates(
        args.checkpoint_dir,
        stem,
        cfg.topk_soup,
    )

    if not top:
        if best_path.exists():
            print(
                "No soup candidates found; keeping calibrated best.pt",
                flush=True,
            )
            print("[5/5] Done:", best_path, flush=True)
            return
        raise RuntimeError("No trained candidates found")

    candidate_objs = [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in top
    ]
    states = [x["ema_model_state"] for x in candidate_objs]
    soup_state = average_state_dicts(states)

    model.load_state_dict(soup_state, strict=True)
    probs_soup, yt, domains = evaluate(
        model,
        val_loader,
        device,
        tta_flip=False,
    )
    soup_robust, soup_m05, _ = robust_selection_score(
        yt, probs_soup, 0.5, domains
    )
    soup_selection = (
        0.50 * soup_m05["Challenge_Avg"]
        + 0.50 * soup_robust
    )

    top1_selection = float(candidate_objs[0]["selection"])
    print(
        "Top1 selection:", top1_selection,
        "| soup selection:", soup_selection,
        flush=True,
    )

    if soup_selection + 0.003 >= top1_selection:
        final_state = soup_state
        final_selection = soup_selection
        final_epoch = max(int(x["epoch"]) for x in candidate_objs)
        source = f"top{len(states)}_weight_soup"
    else:
        final_state = candidate_objs[0]["ema_model_state"]
        final_selection = top1_selection
        final_epoch = int(candidate_objs[0]["epoch"])
        source = "top1_kept_soup_rejected"

    calibrate_and_save(
        backbone=args.backbone,
        seed=args.seed,
        state=final_state,
        model=model,
        val_loader=val_loader,
        device=device,
        best_path=best_path,
        selection=final_selection,
        epoch=final_epoch,
        cfg=cfg,
        source=source,
    )

    print("[5/5] Done:", best_path, flush=True)


if __name__ == "__main__":
    main()

