from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from prototype_par import (
    HybridPrototypePAR,
    cross_domain_prototype_alignment_loss,
    prototype_margin_aux_loss,
    prototype_pair_separation_loss,
)
from strong_par import (
    ATTRIBUTE_NAMES,
    DOMAIN_NAMES,
    MaskedBalancedBCE,
    ModelEMA,
    TrainConfig,
    average_state_dicts,
    load_legacy_teacher,
    response_kd_loss,
    seed_everything,
)
from train_strong_par import (
    evaluate,
    make_loaders,
)
from vapor_par.metrics import (
    evaluate_probs,
    robust_selection_score,
    tune_lodo_thresholds,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/content/UPAR-Challenge-2027"),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/"
            "PedestrianAttributeRecognition/Checkpoints"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sampler-alpha", type=float, default=0.35)

    p.add_argument("--prototype-dim", type=int, default=256)
    p.add_argument("--prototype-temperature", type=float, default=0.20)
    p.add_argument("--prototype-gate-init", type=float, default=0.35)
    p.add_argument("--prototype-ema-momentum", type=float, default=0.95)
    p.add_argument("--prototype-ema-mix", type=float, default=0.25)

    p.add_argument("--prototype-loss-weight", type=float, default=0.22)
    p.add_argument("--prototype-separation-weight", type=float, default=0.010)
    p.add_argument("--domain-alignment-weight", type=float, default=0.020)
    p.add_argument("--prototype-warmup-epochs", type=int, default=1)
    p.add_argument("--alignment-warmup-epochs", type=int, default=2)

    p.add_argument("--backbone-lr", type=float, default=7e-5)
    p.add_argument("--head-lr", type=float, default=2.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.035)

    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    p.add_argument("--no-kd", action="store_true")

    p.add_argument(
        "--init-strong-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional existing strong_convnext_tiny_seed*_best.pt. "
            "Backbone + linear head are copied before prototype training."
        ),
    )

    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def paths_for(args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stem = f"prototype_convnext_tiny_seed{args.seed}"

    return (
        args.checkpoint_dir / f"{stem}_last.pt",
        args.checkpoint_dir / f"{stem}_best.pt",
        stem,
    )


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
        "ema_model_state": {
            k: v.detach().cpu()
            for k, v in model_state.items()
        },
        "metrics_05": metrics_05,
    }
    torch.save(payload, path)


def model_kwargs_from_args(args):
    return {
        "num_attributes": len(ATTRIBUTE_NAMES),
        "prototype_dim": int(args.prototype_dim),
        "prototype_temperature": float(args.prototype_temperature),
        "prototype_gate_init": float(args.prototype_gate_init),
        "prototype_ema_momentum": float(args.prototype_ema_momentum),
        "prototype_ema_mix": float(args.prototype_ema_mix),
    }


def sync_prototype_buffers_to_ema(model, ema):
    """
    ModelEMA already smooths trainable parameters. Prototype EMA buffers are
    themselves moving averages, so they should be copied directly instead of
    being averaged a second time.
    """
    with torch.no_grad():
        for name in (
            "prototype_pos_ema",
            "prototype_neg_ema",
            "prototype_pos_updates",
            "prototype_neg_updates",
        ):
            getattr(ema.module, name).copy_(
                getattr(model, name)
            )


def parameter_groups(model, args):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone.features.") or name.startswith(
            "backbone.classifier.0."
        ):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {
            "params": backbone_params,
            "lr": float(args.backbone_lr),
        },
        {
            "params": head_params,
            "lr": float(args.head_lr),
        },
    ]


def cosine_factor(epoch, total_epochs, floor=0.06):
    if total_epochs <= 1:
        return 1.0

    progress = min(
        max(float(epoch) / float(total_epochs - 1), 0.0),
        1.0,
    )

    return float(
        floor
        + (1.0 - floor)
        * 0.5
        * (1.0 + np.cos(np.pi * progress))
    )


def calibrate_and_save(
    *,
    args,
    state,
    model,
    val_loader,
    device,
    best_path,
    selection,
    epoch,
    source,
):
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # Deployment uses flip-TTA, so calibration must see the same distribution.
    probs, yt, domains = evaluate(
        model,
        val_loader,
        device,
        tta_flip=True,
    )

    thresholds, report = tune_lodo_thresholds(
        yt,
        probs,
        domains,
    )

    calibrated = evaluate_probs(
        yt,
        probs,
        thresholds,
    )

    robust, _, per_domain = robust_selection_score(
        yt,
        probs,
        thresholds,
        domains,
    )

    gate = torch.sigmoid(
        model.prototype_gate_logits.detach().float().cpu()
    ).numpy()

    package = {
        "format": "upar-hybrid-prototype",
        "backbone": "convnext_tiny",
        "seed": int(args.seed),
        "epoch": int(epoch),
        "selection": float(selection),
        "selection_calibrated_robust": float(robust),
        "source": source,
        "inference_model_state": {
            k: v.detach().cpu()
            for k, v in state.items()
        },
        "thresholds": torch.tensor(
            thresholds,
            dtype=torch.float32,
        ),
        "metrics_calibrated": calibrated,
        "lodo_calibration": report,
        "per_domain_calibrated": per_domain,
        "attribute_names": ATTRIBUTE_NAMES,
        "image_height": 288,
        "image_width": 144,
        "prototype_config": model_kwargs_from_args(args),
        "prototype_gate_stats": {
            "min": float(gate.min()),
            "mean": float(gate.mean()),
            "max": float(gate.max()),
        },
    }

    torch.save(
        package,
        best_path,
    )

    print(
        "\nCALIBRATED PROTOTYPE BEST:",
        best_path,
        flush=True,
    )
    print(
        json.dumps(
            calibrated,
            indent=2,
        ),
        flush=True,
    )
    print(
        "Prototype gate stats:",
        package["prototype_gate_stats"],
        flush=True,
    )
    print(
        "LODO:",
        {
            k: report.get(k)
            for k in (
                "global_t",
                "alpha",
                "shift",
                "objective",
            )
        },
        flush=True,
    )

    return package


def main():
    args = parse_args()
    seed_everything(args.seed)

    cfg = TrainConfig(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        epochs=int(args.epochs),
        sampler_alpha=float(args.sampler_alpha),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    last_path, best_path, stem = paths_for(args)

    print("=" * 100, flush=True)
    print(
        "UPAR HYBRID PROTOTYPE TRAINING "
        f"| ConvNeXt-Tiny | seed={args.seed} | device={device}",
        flush=True,
    )
    print("=" * 100, flush=True)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    print("[1/6] Loading data ...", flush=True)
    train_loader, val_loader, pos_weight = make_loaders(
        args,
        cfg,
    )

    print("[2/6] Building hybrid prototype model ...", flush=True)

    model = HybridPrototypePAR(
        **model_kwargs_from_args(args),
        pretrained=True,
    ).to(device)

    # Warm-start from the already strong global ConvNeXt classifier whenever
    # available. Prototype-specific parameters remain new and are learned here.
    if (
        args.init_strong_checkpoint is not None
        and args.init_strong_checkpoint.exists()
        and not (
            args.resume
            and last_path.exists()
        )
    ):
        info = model.load_strong_checkpoint(
            args.init_strong_checkpoint
        )
        print(
            "Warm-started from strong checkpoint:",
            json.dumps(info, indent=2),
            flush=True,
        )

    ema = ModelEMA(
        model,
        decay=cfg.ema_decay,
    )
    ema.module = ema.module.to(device)
    sync_prototype_buffers_to_ema(model, ema)

    teacher = None

    if (
        not args.no_kd
        and args.teacher_checkpoint is not None
    ):
        if args.teacher_checkpoint.exists():
            teacher = load_legacy_teacher(
                args.teacher_checkpoint,
                device,
            )
            print(
                "Low-weight ConvNeXt-Small teacher loaded:",
                args.teacher_checkpoint,
                flush=True,
            )
        else:
            print(
                "Teacher checkpoint missing; training without KD:",
                args.teacher_checkpoint,
                flush=True,
            )

    criterion = MaskedBalancedBCE(
        pos_weight=pos_weight,
        label_smoothing=0.03,
    ).to(device)

    optimizer = torch.optim.AdamW(
        parameter_groups(model, args),
        weight_decay=float(args.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_factor(
            e,
            int(args.epochs),
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    start_epoch = 0
    best_selection = -1e9

    if args.resume and last_path.exists():
        ck = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            ck["model_state"],
            strict=True,
        )
        ema.module.load_state_dict(
            ck["ema_model_state"],
            strict=True,
        )
        optimizer.load_state_dict(
            ck["optimizer"],
        )
        scheduler.load_state_dict(
            ck["scheduler"],
        )

        if "scaler" in ck:
            scaler.load_state_dict(
                ck["scaler"]
            )

        start_epoch = int(ck["epoch"]) + 1
        best_selection = float(
            ck.get(
                "best_selection",
                -1e9,
            )
        )

        print(
            f"Resuming {last_path} "
            f"from epoch {start_epoch + 1}/{args.epochs}",
            flush=True,
        )

    print("[3/6] Training ...", flush=True)

    for epoch in range(
        start_epoch,
        int(args.epochs),
    ):
        model.train()
        epoch_t0 = time.time()

        run = {
            "loss": 0.0,
            "sup": 0.0,
            "proto": 0.0,
            "align": 0.0,
            "sep": 0.0,
            "kd": 0.0,
        }
        steps = 0

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=(
                f"prototype seed{args.seed} "
                f"| epoch {epoch + 1}/{args.epochs}"
            ),
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.25,
            leave=True,
        )

        for batch in progress:
            x = batch["image"].to(
                device,
                non_blocking=True,
            )
            y = batch["target"].to(
                device,
                non_blocking=True,
            )
            valid = batch["valid"].to(
                device,
                non_blocking=True,
            )
            domains = batch["domain"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                out = model(
                    x,
                    return_aux=True,
                )

                logits = out["logits"]

                sup = criterion(
                    logits,
                    y,
                    valid,
                )

                # Prototype branch gets its own weak supervision after a short
                # warm-up so random prototypes do not destabilize epoch 1.
                proto = logits.sum() * 0.0
                proto_weight = 0.0

                if epoch >= int(args.prototype_warmup_epochs):
                    proto = prototype_margin_aux_loss(
                        out["proto_logits"],
                        y,
                        valid,
                        pos_weight=criterion.pos_weight,
                    )

                    proto_ramp = min(
                        1.0,
                        (
                            epoch
                            - int(args.prototype_warmup_epochs)
                            + 1
                        )
                        / 3.0,
                    )
                    proto_weight = (
                        float(args.prototype_loss_weight)
                        * proto_ramp
                    )

                align = logits.sum() * 0.0
                align_weight = 0.0

                if epoch >= int(args.alignment_warmup_epochs):
                    align = cross_domain_prototype_alignment_loss(
                        out["features"],
                        y,
                        valid,
                        domains,
                    )

                    align_ramp = min(
                        1.0,
                        (
                            epoch
                            - int(args.alignment_warmup_epochs)
                            + 1
                        )
                        / 3.0,
                    )

                    align_weight = (
                        float(args.domain_alignment_weight)
                        * align_ramp
                    )

                sep = prototype_pair_separation_loss(
                    out["prototype_pos"],
                    out["prototype_neg"],
                )

                kd = logits.sum() * 0.0
                kd_weight = 0.0

                if (
                    teacher is not None
                    and epoch >= cfg.kd_warmup_epochs
                ):
                    with torch.no_grad():
                        teacher_logits = teacher(x)

                    kd = response_kd_loss(
                        logits,
                        teacher_logits,
                        valid,
                        temperature=cfg.kd_temperature,
                    )

                    kd_ramp = min(
                        1.0,
                        (
                            epoch
                            - cfg.kd_warmup_epochs
                            + 1
                        )
                        / 3.0,
                    )

                    kd_weight = (
                        cfg.kd_weight
                        * kd_ramp
                    )

                loss = (
                    sup
                    + proto_weight * proto
                    + align_weight * align
                    + float(args.prototype_separation_weight) * sep
                    + kd_weight * kd
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}"
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            scaler.step(
                optimizer
            )
            scaler.update()

            # Update stable prototype buffers after the gradient update.
            model.update_prototype_ema(
                out["features"],
                y,
                valid,
            )

            ema.update(model)
            sync_prototype_buffers_to_ema(
                model,
                ema,
            )

            run["loss"] += float(loss.detach())
            run["sup"] += float(sup.detach())
            run["proto"] += float(proto.detach())
            run["align"] += float(align.detach())
            run["sep"] += float(sep.detach())
            run["kd"] += float(kd.detach())
            steps += 1

            gate_mean = float(
                torch.sigmoid(
                    model.prototype_gate_logits.detach()
                ).mean()
            )

            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                sup=f"{float(sup.detach()):.4f}",
                proto=f"{float(proto.detach()):.4f}",
                align=f"{float(align.detach()):.4f}",
                gate=f"{gate_mean:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                refresh=False,
            )

        progress.close()
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} finished "
            f"in {(time.time() - epoch_t0) / 60:.1f} min",
            flush=True,
        )

        print(
            "[4/6] Source-domain validation ...",
            flush=True,
        )

        probs, yt, domains_np = evaluate(
            ema.module,
            val_loader,
            device,
            tta_flip=False,
        )

        m05 = evaluate_probs(
            yt,
            probs,
            0.5,
        )

        robust, _, per_domain = robust_selection_score(
            yt,
            probs,
            0.5,
            domains_np,
        )

        selection = (
            0.50 * m05["Challenge_Avg"]
            + 0.50 * robust
        )

        gate = torch.sigmoid(
            ema.module.prototype_gate_logits.detach().float().cpu()
        )

        summary = {
            "epoch": epoch + 1,
            "seed": args.seed,
            "selection": selection,
            "metrics_05": m05,
            "robust_05": robust,
            "per_domain_05": {
                DOMAIN_NAMES[d]: m["Challenge_Avg"]
                for d, m in per_domain.items()
            },
            "prototype_gate": {
                "min": float(gate.min()),
                "mean": float(gate.mean()),
                "max": float(gate.max()),
            },
            "train": {
                k: v / max(steps, 1)
                for k, v in run.items()
            },
        }

        print(
            json.dumps(
                summary,
                indent=2,
            ),
            flush=True,
        )

        last_payload = {
            "format": "upar-hybrid-prototype-last",
            "epoch": epoch,
            "seed": args.seed,
            "best_selection": max(
                best_selection,
                selection,
            ),
            "model_state": model.state_dict(),
            "ema_model_state": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics_05": m05,
            "robust_05": robust,
            "attribute_names": ATTRIBUTE_NAMES,
            "image_height": 288,
            "image_width": 144,
            "prototype_config": model_kwargs_from_args(args),
        }

        torch.save(
            last_payload,
            last_path,
        )

        candidate = (
            args.checkpoint_dir
            / (
                f"{stem}_ep{epoch + 1:02d}"
                f"_score{selection:.6f}.pt"
            )
        )

        save_candidate(
            candidate,
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

            print(
                "New best prototype source selection:",
                best_selection,
                flush=True,
            )

            calibrate_and_save(
                args=args,
                state=ema.module.state_dict(),
                model=model,
                val_loader=val_loader,
                device=device,
                best_path=best_path,
                selection=selection,
                epoch=epoch,
                source="single_best",
            )

        print(
            "Top soup candidates:",
            [p.name for p in top],
            flush=True,
        )

    print(
        "[5/6] Building top-k prototype weight soup ...",
        flush=True,
    )

    top = prune_candidates(
        args.checkpoint_dir,
        stem,
        cfg.topk_soup,
    )

    if not top:
        if best_path.exists():
            print(
                "No soup candidates; keeping calibrated best.pt",
                flush=True,
            )
            print(
                "[6/6] Done:",
                best_path,
                flush=True,
            )
            return

        raise RuntimeError(
            "No prototype candidates found"
        )

    candidate_objs = [
        torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )
        for p in top
    ]

    states = [
        x["ema_model_state"]
        for x in candidate_objs
    ]

    soup_state = average_state_dicts(
        states
    )

    model.load_state_dict(
        soup_state,
        strict=True,
    )

    probs_soup, yt, domains_np = evaluate(
        model,
        val_loader,
        device,
        tta_flip=False,
    )

    soup_robust, soup_m05, _ = robust_selection_score(
        yt,
        probs_soup,
        0.5,
        domains_np,
    )

    soup_selection = (
        0.50 * soup_m05["Challenge_Avg"]
        + 0.50 * soup_robust
    )

    top1_selection = float(
        candidate_objs[0]["selection"]
    )

    print(
        "Top1 selection:",
        top1_selection,
        "| soup selection:",
        soup_selection,
        flush=True,
    )

    # Quality gate: soup is optional.
    if soup_selection + 0.003 >= top1_selection:
        final_state = soup_state
        final_selection = soup_selection
        final_epoch = max(
            int(x["epoch"])
            for x in candidate_objs
        )
        source = (
            f"top{len(states)}_weight_soup"
        )
    else:
        final_state = candidate_objs[0][
            "ema_model_state"
        ]
        final_selection = top1_selection
        final_epoch = int(
            candidate_objs[0]["epoch"]
        )
        source = (
            "top1_kept_soup_rejected"
        )

    calibrate_and_save(
        args=args,
        state=final_state,
        model=model,
        val_loader=val_loader,
        device=device,
        best_path=best_path,
        selection=final_selection,
        epoch=final_epoch,
        source=source,
    )

    print(
        "[6/6] Done:",
        best_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
