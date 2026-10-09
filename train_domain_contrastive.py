from __future__ import annotations

import argparse
import math
import pandas as pd
from torch.utils.data import DataLoader, Subset
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from domain_contrastive import (
    DomainMixBatchSampler,
    cross_domain_supcon,
    domain_positive_negative_alignment,
)
from prototype_par import (
    HybridPrototypePAR,
    cross_domain_prototype_alignment_loss,
    prototype_margin_aux_loss,
    prototype_pair_separation_loss,
    hard_negative_false_positive_loss,
    prototype_gate_anchor_loss,
)
from strong_par import (
    ATTRIBUTE_NAMES,
    DOMAIN_NAMES,
    MaskedBalancedBCE,
    ImageResolver, UPARDataset, build_train_transform, build_eval_transform,
    dataset_pos_weight,
    ModelEMA,
    TrainConfig,
    average_state_dicts,
    load_legacy_teacher,
    response_kd_loss,
    seed_everything,
)
from train_strong_par import (
    evaluate,
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
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sampler-alpha", type=float, default=0.35)

    p.add_argument("--prototype-dim", type=int, default=256)
    p.add_argument("--prototype-temperature", type=float, default=0.20)
    p.add_argument("--prototype-gate-init", type=float, default=0.35)
    p.add_argument("--prototype-gate-min", type=float, default=0.08)
    p.add_argument("--prototype-gate-max", type=float, default=0.92)
    p.add_argument("--prototype-ema-momentum", type=float, default=0.95)
    p.add_argument("--prototype-ema-mix", type=float, default=0.25)

    p.add_argument("--prototype-loss-weight", type=float, default=0.08)
    p.add_argument("--prototype-separation-weight", type=float, default=0.010)
    p.add_argument("--domain-alignment-weight", type=float, default=0.020)
    p.add_argument("--prototype-warmup-epochs", type=int, default=1)
    p.add_argument("--alignment-warmup-epochs", type=int, default=2)

    p.add_argument("--pos-weight-cap", type=float, default=2.5)
    p.add_argument("--hard-negative-weight", type=float, default=0.0)
    p.add_argument("--hard-negative-gamma", type=float, default=2.0)
    p.add_argument("--hard-negative-min-prob", type=float, default=0.35)
    p.add_argument("--gate-anchor-target", type=float, default=0.35)
    p.add_argument("--gate-anchor-weight", type=float, default=0.0)

    p.add_argument("--backbone-lr", type=float, default=8e-5)
    p.add_argument("--head-lr", type=float, default=2.4e-4)
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

    p.add_argument("--init-prototype-checkpoint", type=Path, default=None,
                   help="Frozen original Prototype checkpoint for warm-start and stable-reference KD")
    p.add_argument("--reference-kd-weight", type=float, default=0.08)
    p.add_argument("--reference-temperature", type=float, default=1.5)
    p.add_argument("--domain-robust-weight", type=float, default=0.00)
    p.add_argument("--allow-from-scratch", action="store_true",
                   help="Not recommended: run without a pretrained Prototype checkpoint")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--heldout-domain", type=int, default=-1,
                   help="-1=train on all source domains, 0/1/2=strict LODO fold")
    p.add_argument("--steps-per-epoch", type=int, default=900)
    p.add_argument("--contrastive-weight", type=float, default=0.06)
    p.add_argument("--contrastive-temp", type=float, default=0.18)
    p.add_argument("--contrastive-warmup-epochs", type=int, default=2)
    p.add_argument("--contrastive-max-attributes", type=int, default=16)
    p.add_argument("--trial-tag", type=str, default="full")
    p.add_argument("--report-dir", type=Path, default=None)
    return p.parse_args()


def paths_for(args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.heldout_domain < 0:
        stem = f"prototype_contrastive_convnext_tiny_seed{args.seed}"
    else:
        tag = ''.join(c for c in args.trial_tag if c.isalnum() or c in "_-" )
        stem = f"prototype_lodo_seed{args.seed}_heldout{args.heldout_domain}_{tag}"


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
        "prototype_gate_min": float(args.prototype_gate_min),
        "prototype_gate_max": float(args.prototype_gate_max),
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
        "training_method": "fresh_student_cross_domain_supcon",
        "domain_objective": {
            "contrastive_weight": float(args.contrastive_weight),
            "alignment_weight": float(args.domain_alignment_weight),
            "contrastive_temperature": float(args.contrastive_temp),
            "heldout_domain": int(args.heldout_domain),
            "teacher_weight": float(args.reference_kd_weight),
        },
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



def make_domain_loaders(args, cfg):
    """TRUE LODO: excludes held-out domain from *training annotations*.

    IMPORTANT: LODO folds NEVER load any teacher or initialization checkpoint
    trained on all three source domains. Only public ImageNet pretraining is used.
    """
    data_root = args.repo_root / "data"
    train_csv = data_root / "annotations/task1/train/gt.csv"
    val_csv = data_root / "annotations/task1/val/gt.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("UPAR annotations missing; run prepare_data.py")
    resolver = ImageResolver(data_root, args.repo_root)
    train_data = UPARDataset(
        pd.read_csv(train_csv), resolver,
        build_train_transform(cfg.image_height, cfg.image_width),
    )
    val_data = UPARDataset(
        pd.read_csv(val_csv), resolver,
        build_eval_transform(cfg.image_height, cfg.image_width),
    )
    all_domains = np.asarray(train_data.domains, dtype=np.int64)
    if args.heldout_domain >= 0:
        if args.heldout_domain not in (0, 1, 2):
            raise ValueError('heldout-domain must be -1,0,1,2')
        tr_indices = np.flatnonzero((all_domains >= 0) & (all_domains != args.heldout_domain))
        va_indices = np.flatnonzero(val_data.domains == args.heldout_domain)
        if not len(va_indices):
            raise ValueError('Held-out validation domain has no examples')
    else:
        tr_indices = np.flatnonzero(all_domains >= 0)
        va_indices = np.arange(len(val_data), dtype=np.int64)
    subset_domains = all_domains[tr_indices]
    print('STRICT domain train counts:',
          {DOMAIN_NAMES[int(i)]:int((subset_domains==i).sum()) for i in np.unique(subset_domains)},
          '| validation:',
          {DOMAIN_NAMES[int(i)]:int((val_data.domains[va_indices]==i).sum())
           for i in np.unique(val_data.domains[va_indices])}, flush=True)
    if args.heldout_domain >= 0:
        assert not np.any(subset_domains == args.heldout_domain), 'Domain leakage!'
    sampler = DomainMixBatchSampler(
        subset_domains, batch_size=int(args.batch_size),
        steps_per_epoch=int(args.steps_per_epoch),
        alpha=float(args.sampler_alpha), seed=int(args.seed), min_per_domain=8,
    )
    train_loader = DataLoader(
        Subset(train_data, tr_indices.tolist()), batch_sampler=sampler,
        num_workers=int(args.num_workers), pin_memory=True,
        persistent_workers=int(args.num_workers)>0,
    )
    val_loader = DataLoader(
        Subset(val_data, va_indices.tolist()),
        batch_size=max(64, int(args.batch_size)), shuffle=False,
        num_workers=int(args.num_workers), pin_memory=True,
        persistent_workers=int(args.num_workers)>0,
    )
    pos_weight = dataset_pos_weight(train_data.targets[tr_indices])
    return train_loader, val_loader, pos_weight


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
        "UPAR PROTOTYPE DOMAIN-ROBUST FINETUNING "
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
    train_loader, val_loader, pos_weight = make_domain_loaders(
        args,
        cfg,
    )
    pos_weight = np.minimum(
        np.asarray(pos_weight, dtype=np.float32),
        float(args.pos_weight_cap),
    )
    print(
        "Precision-oriented pos_weight range:",
        float(pos_weight.min()),
        float(pos_weight.max()),
        flush=True,
    )

    # Student is NEW: ImageNet pretrained ConvNeXt + new attribute queries,
    # new projection and new prototype bank. This is NOT a tiny fine-tune of
    # the old Prototype checkpoint.
    if args.heldout_domain >= 0 and args.init_prototype_checkpoint is not None:
        raise ValueError('LODO cannot use full-source Prototype teacher (data leakage)')
    if args.heldout_domain >= 0 and args.init_strong_checkpoint is not None:
        raise ValueError('LODO cannot use full-source Strong checkpoint (data leakage)')
    if args.heldout_domain >= 0 and args.teacher_checkpoint is not None:
        raise ValueError('LODO cannot use a full-source external teacher')
    initial_checkpoint = None
    initial_state = None
    if args.heldout_domain < 0:
        if args.init_prototype_checkpoint is None or not args.init_prototype_checkpoint.exists():
            raise FileNotFoundError('Full-source training requires old Prototype teacher checkpoint')
        initial_checkpoint = torch.load(args.init_prototype_checkpoint,
                                        map_location='cpu', weights_only=False)
        for key in ('inference_model_state','ema_model_state','model_state'):
            if key in initial_checkpoint and initial_checkpoint[key] is not None:
                initial_state = initial_checkpoint[key]
                break
        if initial_state is None:
            raise RuntimeError('Prototype teacher checkpoint has no model weights')

    print('[2/6] New ImageNet-initialized student (fresh prototype representations)', flush=True)
    # HybridPrototypePAR's legacy constructor silently falls back to random
    # initialization if the download fails. Never allow such a silent fallback
    # in domain-generalization research: require actual ImageNet weights.
    from torchvision.models import ConvNeXt_Tiny_Weights
    try:
        ConvNeXt_Tiny_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    except Exception as exc:
        raise RuntimeError(
            'ImageNet-pretrained ConvNeXt-Tiny weights are REQUIRED; '
            'check Colab internet/torchvision cache before training.'
        ) from exc
    model = HybridPrototypePAR(**model_kwargs_from_args(args), pretrained=True).to(device)
    print('Student initialization: ImageNet only, NOT old Prototype weights', flush=True)

    ema = ModelEMA(
        model,
        decay=cfg.ema_decay,
    )
    ema.module = ema.module.to(device)
    sync_prototype_buffers_to_ema(model, ema)

    reference = None
    if args.heldout_domain < 0 and initial_state is not None and float(args.reference_kd_weight) > 0:
        teacher_cfg = dict(initial_checkpoint.get('prototype_config', {}))
        teacher_cfg.pop('pretrained', None)
        reference = HybridPrototypePAR(pretrained=False, **teacher_cfg)
        reference.load_state_dict(initial_state, strict=True)
        reference = reference.to(device).eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        print('Frozen old Prototype teacher used for response KD ONLY',flush=True)
    # No other teacher: the old Prototype is our single reference.
    teacher = None

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

    objective_config = {
        'heldout_domain': int(args.heldout_domain),
        'contrastive_weight': float(args.contrastive_weight),
        'domain_alignment_weight': float(args.domain_alignment_weight),
        'epochs': int(args.epochs),
        'steps_per_epoch': int(args.steps_per_epoch),
        'prototype_dim': int(args.prototype_dim),
    }
    start_epoch = 0
    best_selection = -1e9

    if args.resume and last_path.exists():
        ck = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False,
        )

        saved_objective = ck.get('objective_config')
        if saved_objective != objective_config:
            raise RuntimeError(
                'Cannot resume checkpoint trained with DIFFERENT domain-loss '
                f'configuration. checkpoint={saved_objective}, requested={objective_config}. '
                'Use a fresh checkpoint directory or retain the prior settings.'
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
        train_loader.batch_sampler.set_epoch(epoch)
        epoch_t0 = time.time()

        run = {
            "loss": 0.0,
            "sup": 0.0,
            "proto": 0.0,
            "align": 0.0,
            "supcon": 0.0,
            "sep": 0.0,
            "kd": 0.0,
            "hardneg": 0.0,
            "gate_anchor": 0.0,
            "reference_kd": 0.0,
            "domain_robust": 0.0,
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
                    align = domain_positive_negative_alignment(
                        out['features'], y, valid, domains,
                        min_per_cell=2,
                        max_attributes=int(args.contrastive_max_attributes),
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

                supcon = logits.sum() * 0.0
                contrastive_weight = 0.0
                if epoch >= int(args.contrastive_warmup_epochs):
                    supcon = cross_domain_supcon(
                        out['features'], y, valid, domains,
                        temperature=float(args.contrastive_temp),
                        max_per_cell=8,
                        max_attributes=int(args.contrastive_max_attributes),
                    )
                    contrastive_weight = float(args.contrastive_weight) * min(
                        1.0, (epoch-int(args.contrastive_warmup_epochs)+1)/3.0
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

                hardneg = hard_negative_false_positive_loss(
                    logits,
                    y,
                    valid,
                    gamma=float(args.hard_negative_gamma),
                    min_probability=float(args.hard_negative_min_prob),
                )

                gate_anchor = prototype_gate_anchor_loss(
                    out["prototype_gate"],
                    target=float(args.gate_anchor_target),
                )

                # Avoid over-reliance on the largest source domain.
                # This is a small worst-domain *excess* term; it is only
                # computed among domains actually present in this minibatch.
                source_losses = []
                for source_id in range(len(DOMAIN_NAMES)):
                    rows = (domains == source_id)
                    if int(rows.sum().item()) >= 2 and bool(valid[rows].sum().item() > 0):
                        source_losses.append(
                            criterion(logits[rows], y[rows], valid[rows])
                        )
                domain_robust = (torch.stack(source_losses).max()
                                 - torch.stack(source_losses).mean()) if len(source_losses) >= 2                                  else logits.sum() * 0.0

                reference_kd = logits.sum() * 0.0
                if reference is not None:
                    with torch.no_grad():
                        ref_logits = reference(x)
                    reference_kd = response_kd_loss(
                        logits, ref_logits, valid,
                        temperature=float(args.reference_temperature),
                    )

                loss = (
                    sup
                    + proto_weight * proto
                    + align_weight * align
                    + contrastive_weight * supcon
                    + float(args.prototype_separation_weight) * sep
                    + kd_weight * kd
                    + float(args.hard_negative_weight) * hardneg
                    + float(args.gate_anchor_weight) * gate_anchor
                    + float(args.reference_kd_weight) * reference_kd
                    + float(args.domain_robust_weight) * domain_robust
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
            run["supcon"] += float(supcon.detach())
            run["sep"] += float(sep.detach())
            run["kd"] += float(kd.detach())
            run["hardneg"] += float(hardneg.detach())
            run["gate_anchor"] += float(gate_anchor.detach())
            run["reference_kd"] += float(reference_kd.detach())
            run["domain_robust"] += float(domain_robust.detach())
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
                supcon=f"{float(supcon.detach()):.3f}",
                gate=f"{gate_mean:.2f}",
                hn=f"{float(hardneg.detach()):.3f}",
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
            "objective_config": objective_config,
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

            if args.heldout_domain >= 0:
                torch.save({
                    'epoch':int(epoch), 'seed':int(args.seed),
                    'selection':float(selection), 'metrics_05':m05,
                    'prototype_config':model_kwargs_from_args(args),
                    'inference_model_state':{
                        k:v.detach().cpu().clone()
                        for k,v in ema.module.state_dict().items()},
                    'heldout_domain':int(args.heldout_domain),
                },best_path)
            else:
                calibrate_and_save(
                    args=args,state=ema.module.state_dict(),model=ema.module,
                    val_loader=val_loader,device=device,best_path=best_path,
                    selection=selection,epoch=epoch,source='single_best',
                )

        print(
            "Top soup candidates:",
            [p.name for p in top],
            flush=True,
        )

    if args.heldout_domain >= 0:
        if args.report_dir is None:
            raise ValueError('--report-dir is required for LODO trials')
        args.report_dir.mkdir(parents=True,exist_ok=True)
        report = {
            'heldout_domain':int(args.heldout_domain),
            'heldout_name':DOMAIN_NAMES[int(args.heldout_domain)],
            'seed':int(args.seed),
            'contrastive_weight':float(args.contrastive_weight),
            'alignment_weight':float(args.domain_alignment_weight),
            'best_selection':float(best_selection),
            'best_checkpoint':str(best_path),
            'teacher_used':False,
            'image_pretraining':'ImageNet',
            'training_domain_leakage':False,
        }
        report_path = args.report_dir / f'{stem}.json'
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('LODO RESULT:',json.dumps(report,indent=2),flush=True)
        return
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
