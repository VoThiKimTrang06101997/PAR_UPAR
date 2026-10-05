from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from distill_par import (
    ATTRIBUTE_NAMES, DOMAIN_NAMES, ConvNeXtPAR, ImageResolver, MaskedAsymmetricLoss,
    ModelEMA, TrainConfig, UPARDataset, build_eval_transform, build_train_transform,
    confidence_distillation_loss, domain_loss, exclusivity_regularizer,
    feature_distillation_loss, make_domain_balanced_sampler, seed_everything,
)
from vapor_par.metrics import evaluate_probs, robust_selection_score, tune_lodo_thresholds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["teacher","student"], required=True)
    p.add_argument("--repo-root", type=Path, default=Path("/content/UPAR-Challenge-2027"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    p.add_argument("--init-student", type=Path, default=None)
    return p.parse_args()


def paths_for(args, cfg):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "teacher":
        stem = f"distill_teacher_convnext_small_seed{args.seed}"
    else:
        stem = f"distill_student_convnext_tiny_seed{args.seed}"
    return args.checkpoint_dir / f"{stem}_last.pt", args.checkpoint_dir / f"{stem}_best.pt"


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    probs, ys, ds = [], [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
            logits = model(x)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
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
        raise FileNotFoundError("Official task1 train/val annotations not found. Run prepare_data.py first.")
    train_df, val_df = pd.read_csv(train_csv), pd.read_csv(val_csv)
    resolver = ImageResolver(data_root, args.repo_root)
    train_ds = UPARDataset(train_df, resolver, build_train_transform(cfg.image_height, cfg.image_width))
    val_ds = UPARDataset(val_df, resolver, build_eval_transform(cfg.image_height, cfg.image_width))
    sampler = make_domain_balanced_sampler(train_ds.domains, args.seed)
    bs = args.batch_size or cfg.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers>0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(bs,64), shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers>0,
    )
    print("Train:", train_df.shape, "Val:", val_df.shape)
    print("Train domain counts:", {DOMAIN_NAMES[d]: int((train_ds.domains==d).sum()) for d in range(3)})
    return train_loader, val_loader


def state_for_inference(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("domain_head.")}


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = TrainConfig(num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args, cfg)
    last_path, best_path = paths_for(args, cfg)

    if args.stage == "teacher":
        model = ConvNeXtPAR("small", pretrained=True, mixstyle=True).to(device)
        teacher = None
        epochs = args.epochs or cfg.teacher_epochs
        lr = cfg.lr_teacher
    else:
        model = ConvNeXtPAR("tiny", pretrained=True, mixstyle=True).to(device)
        teacher_path = args.teacher_checkpoint or (args.checkpoint_dir / "distill_teacher_convnext_small_seed42_best.pt")
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
        tckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher = ConvNeXtPAR("small", pretrained=False, mixstyle=False)
        teacher.load_state_dict(tckpt["ema_model_state"], strict=True)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        if args.init_student and args.init_student.exists():
            obj = torch.load(args.init_student, map_location="cpu", weights_only=False)
            st = obj.get("ema_model_state") or obj.get("model_state") or obj
            missing, unexpected = model.load_state_dict(st, strict=False)
            print("Student warm start:", args.init_student, "missing", len(missing), "unexpected", len(unexpected))
        epochs = args.epochs or cfg.student_epochs
        lr = cfg.lr_student

    criterion = MaskedAsymmetricLoss()
    ema = ModelEMA(model, cfg.ema_decay)
    ema.module = ema.module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs,1), eta_min=lr*0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    start_epoch, best_selection = 0, -1e9

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        ema.module.load_state_dict(ckpt["ema_model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_selection = float(ckpt.get("best_selection", -1e9))
        print("Resuming", last_path, "from epoch", start_epoch)

    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"loss":0.0,"sup":0.0,"kd":0.0,"feat":0.0,"domain":0.0,"group":0.0}
        steps = 0
        prog = tqdm(train_loader, desc=f"{args.stage} seed{args.seed} {epoch+1}/{epochs}")
        for step,batch in enumerate(prog):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            domains = batch["domain"].to(device, non_blocking=True)
            progress = (epoch + step/max(len(train_loader),1)) / max(epochs,1)
            grl = min(0.25, 0.25 * (2.0/(1.0+math.exp(-8*progress))-1.0))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                logits, dlogits, feat = model(x, grl_lambda=grl, return_domain=True)
                sup = criterion(logits, y, valid)
                dl = domain_loss(dlogits, domains)
                grp = exclusivity_regularizer(logits)
                kd = logits.sum()*0.0
                feat_kd = logits.sum()*0.0
                if teacher is not None:
                    with torch.no_grad():
                        tlogits, tfeat = teacher(x, return_features=True)
                    kd = confidence_distillation_loss(logits, tlogits, valid, cfg.temperature)
                    feat_kd = feature_distillation_loss(feat, tfeat)
                    loss = sup + cfg.kd_weight*kd + cfg.feature_kd_weight*feat_kd + cfg.domain_loss_weight*dl + cfg.group_weight*grp
                else:
                    loss = sup + cfg.domain_loss_weight*dl + cfg.group_weight*grp
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); ema.update(model)
            vals = {"loss":loss,"sup":sup,"kd":kd,"feat":feat_kd,"domain":dl,"group":grp}
            for k,v in vals.items(): running[k] += float(v.detach().item())
            steps += 1
            prog.set_postfix(loss=f"{loss.item():.4f}", sup=f"{sup.item():.4f}", kd=f"{kd.item():.4f}")
        scheduler.step()

        probs, yt, domains_np = evaluate(ema.module, val_loader, device)
        m05 = evaluate_probs(yt, probs, 0.5)
        robust, _, per_domain = robust_selection_score(yt, probs, 0.5, domains_np)
        selection = 0.55*m05["Challenge_Avg"] + 0.45*robust
        summary = {
            "stage":args.stage,"epoch":epoch+1,"seed":args.seed,
            "selection":selection,"metrics_05":m05,"robust_05":robust,
            "per_domain_05":{DOMAIN_NAMES.get(d,d) if isinstance(DOMAIN_NAMES,dict) else DOMAIN_NAMES[d]:m["Challenge_Avg"] for d,m in per_domain.items()},
            "train":{k:v/max(steps,1) for k,v in running.items()},
        }
        print(json.dumps(summary, indent=2))
        payload = {
            "format":"upar-distill-convnext-v1", "stage":args.stage,
            "backbone":"convnext_small" if args.stage=="teacher" else "convnext_tiny",
            "epoch":epoch,"seed":args.seed,"best_selection":max(best_selection,selection),
            "model_state":model.state_dict(),"ema_model_state":ema.module.state_dict(),
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
            "metrics_05":m05,"robust_05":robust,"per_domain_05":per_domain,
            "attribute_names":ATTRIBUTE_NAMES,"image_height":cfg.image_height,"image_width":cfg.image_width,
            "config":vars(cfg),
        }
        torch.save(payload,last_path)
        if selection > best_selection:
            best_selection = float(selection)
            torch.save(payload,best_path)
            print("Saved BEST:",best_path)

    # Re-evaluate best EMA and calibrate with leave-one-domain-out transfer.
    best = torch.load(best_path,map_location="cpu",weights_only=False)
    model.load_state_dict(best["ema_model_state"],strict=True)
    model = model.to(device).eval()
    probs, yt, domains_np = evaluate(model,val_loader,device)
    thresholds, lodo = tune_lodo_thresholds(yt, probs, domains_np)
    calibrated = evaluate_probs(yt, probs, thresholds)
    best["thresholds"] = torch.tensor(thresholds,dtype=torch.float32)
    best["lodo_calibration"] = lodo
    best["metrics_calibrated"] = calibrated
    best["inference_model_state"] = state_for_inference(model)
    torch.save(best,best_path)
    print("\nFINAL",args.stage,"CHECKPOINT:",best_path)
    print("Calibrated:",json.dumps(calibrated,indent=2))
    print("LODO params:",{k:lodo.get(k) for k in ("global_t","alpha","shift","objective")})
    print("Threshold range:",float(thresholds.min()),float(thresholds.max()))


if __name__ == "__main__":
    main()
