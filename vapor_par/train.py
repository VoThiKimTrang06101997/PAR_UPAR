from pathlib import Path
import math
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from .config import Config, ATTRIBUTE_NAMES
from .data import (
    UPARAttributeDataset,
    TrainCollator,
    EvalCollator,
    ResumableEpochBatchSampler,
)
from .model import VAPORPAR
from .losses import total_vapor_loss
from .metrics import evaluate_probs, tune_attribute_thresholds
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_payload,
    apply_checkpoint_payload,
)
from .calibration import fit_vector_scaling, apply_calibration
from .utils import seed_everything, move_inputs, append_csv, get_amp_dtype


def _build_optimizer(model, cfg):
    head, backbone = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith("vlm.") else head).append(p)
    groups = []
    if head:
        groups.append({"params": head, "lr": cfg.lr_head})
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _positive_prior(ds):
    return torch.from_numpy(ds.targets.mean(0).astype("float32"))


def _find_resume_checkpoint(cfg):
    """Return the checkpoint that is furthest through training.

    Both resume_step.pt and last.pt may exist after an interrupted/partially
    completed Colab session. Do not blindly prefer one filename: inspect both
    and choose the state with the largest (global_step, epoch, next_batch_idx).
    """
    ckpt_dir = Path(cfg.checkpoint_dir)
    candidates = [
        ckpt_dir / "resume_step.pt",
        ckpt_dir / "last.pt",
    ]
    scored = []

    for path in candidates:
        if not path.exists():
            continue
        try:
            ck = load_checkpoint_payload(path, device="cpu")
            score = (
                int(ck.get("global_step", -1)),
                int(ck.get("epoch", -1)),
                int(ck.get("next_batch_idx", -1)),
            )
            scored.append((score, path))
        except Exception as exc:
            print(f"[WARN] Could not inspect checkpoint {path}: {exc}")

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]
    print(
        f"[AUTO-RESUME] Selected {best_path.name} "
        f"(global_step={best_score[0]}, epoch={best_score[1]}, "
        f"next_batch={best_score[2]})"
    )
    return best_path


def _apply_resume_critical_config(cfg, ckpt):
    """Preserve sample ordering/model shape when resuming an existing run.

    In particular, changing batch_size after a mid-epoch checkpoint changes what
    `batch=4000` means. Therefore the checkpoint value wins for resume-critical fields.
    """
    old = ckpt.get("cfg") or {}
    locked = (
        "model_id",
        "num_attributes",
        "num_heads",
        "seed",
        "batch_size",
        "accumulation_steps",
    )
    for key in locked:
        if key in old and hasattr(cfg, key):
            old_value = old[key]
            new_value = getattr(cfg, key)
            if old_value != new_value:
                print(
                    f"[RESUME CONFIG] {key}: requested={new_value!r} -> "
                    f"checkpoint={old_value!r} (checkpoint value kept)"
                )
                setattr(cfg, key, old_value)


def _checkpoint_data_state(train_ds, batch_sampler, cfg):
    return {
        "dataset_size": len(train_ds),
        "batch_size": int(cfg.batch_size),
        "total_batches": int(batch_sampler.total_batches),
        "seed": int(cfg.seed),
    }


@torch.no_grad()
def evaluate(model, loader, device, calibration=None, desc="Evaluation"):
    model.eval()
    all_logits, all_targets = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        inputs = move_inputs(batch["inputs"], device)
        out = model(inputs)
        logits = apply_calibration(out["logits"], calibration)
        all_logits.append(logits.float().cpu())
        all_targets.append(batch["targets"].float().cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.sigmoid(logits)
    return logits, targets, probs


def run_training(cfg: Config, resume=True):
    cfg.ensure_dirs()

    ckpt_dir = Path(cfg.checkpoint_dir)
    res_dir = Path(cfg.results_dir)
    resume_path = ckpt_dir / "resume_step.pt"
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"

    # Read resume metadata BEFORE creating the DataLoader. This allows us to restore
    # the original batch size/seed and construct a sampler that starts directly at
    # the saved batch instead of loading batches 0..resume_batch-1 again.
    chosen = _find_resume_checkpoint(cfg) if resume else None
    resume_ckpt = None
    if chosen is not None:
        resume_ckpt = load_checkpoint_payload(chosen, device="cpu")
        _apply_resume_critical_config(cfg, resume_ckpt)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        cache_dir=(cfg.hf_cache_dir or None),
    )
    train_ds = UPARAttributeDataset(cfg.train_csv, cfg.data_dir, cfg.max_train_samples)
    val_ds = UPARAttributeDataset(cfg.val_csv, cfg.data_dir, cfg.max_val_samples)

    batch_sampler = ResumableEpochBatchSampler(
        train_ds,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        drop_last=False,
    )
    full_train_batches = batch_sampler.total_batches

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=TrainCollator(processor),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=EvalCollator(processor, True),
    )

    model = VAPORPAR(cfg).to(device)
    model.initialize_prompts(processor, device)
    pos_prior = _positive_prior(train_ds).to(device)

    start_epoch, resume_batch, global_step, best_score = 0, 0, 0, -1.0
    thresholds, calibration = None, None

    # Recreate Stage-C trainable topology before optimizer construction.
    if resume_ckpt is not None:
        resume_epoch_hint = int(resume_ckpt.get("epoch", 0))
        if (
            resume_epoch_hint >= cfg.unfreeze_start_epoch
            and cfg.unfreeze_last_n_vision_blocks > 0
        ):
            model.unfreeze_last_vision_blocks(cfg.unfreeze_last_n_vision_blocks)

    optimizer = _build_optimizer(model, cfg)
    opt_steps_per_epoch = math.ceil(full_train_batches / cfg.accumulation_steps)
    total_opt_steps = opt_steps_per_epoch * cfg.epochs
    warmup = int(cfg.warmup_ratio * total_opt_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_opt_steps)
    amp_dtype = get_amp_dtype()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.amp and device.type == "cuda" and amp_dtype == torch.float16),
    )

    if resume_ckpt is not None:
        apply_checkpoint_payload(
            resume_ckpt,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            restore_rng=True,
        )
        start_epoch = int(resume_ckpt.get("epoch", 0))
        resume_batch = int(resume_ckpt.get("next_batch_idx", 0))
        global_step = int(resume_ckpt.get("global_step", 0))
        best_score = float(resume_ckpt.get("best_score", -1.0))
        thresholds = resume_ckpt.get("thresholds")
        calibration = resume_ckpt.get("calibration")

        old_data = resume_ckpt.get("data_state") or {}
        if old_data.get("dataset_size") not in (None, len(train_ds)):
            raise RuntimeError(
                "Dataset size changed since checkpoint: "
                f"checkpoint={old_data.get('dataset_size')} current={len(train_ds)}. "
                "Exact mid-epoch resume is unsafe."
            )

        # Backward-compatible with older checkpoint version used in the screenshot.
        if resume_batch >= full_train_batches:
            advance = resume_batch // full_train_batches
            start_epoch += advance
            resume_batch = resume_batch % full_train_batches

        pct = 100.0 * resume_batch / max(full_train_batches, 1)
        print(
            f"[RESUME] {chosen} -> epoch={start_epoch}, "
            f"next_batch={resume_batch}/{full_train_batches} ({pct:.2f}%), "
            f"global_step={global_step}"
        )
        print(
            "[RESUME] Direct sampler seek is ENABLED: already-completed batches "
            "will NOT be decoded/preprocessed again."
        )

    epoch_fields = [
        "epoch", "resumed_from_batch", "lr", "train_loss", "mA",
        "instance_precision", "instance_recall", "instance_f1",
        "challenge_score", "best_score", "seconds",
    ]
    step_fields = [
        "epoch", "batch", "global_step", "loss", "bal", "sem", "dg",
        "anchor", "prompt", "rel", "onto", "cons", "featcons", "lr",
    ]

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start_batch = resume_batch if epoch == start_epoch else 0
        batch_sampler.set_epoch(epoch, start_batch=epoch_start_batch)

        if (
            epoch == cfg.unfreeze_start_epoch
            and cfg.unfreeze_last_n_vision_blocks > 0
            and not any(p.requires_grad for p in model.vlm.vision_model.parameters())
        ):
            model.unfreeze_last_vision_blocks(cfg.unfreeze_last_n_vision_blocks)
            optimizer = _build_optimizer(model, cfg)
            remaining = max(1, total_opt_steps - global_step)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                min(warmup, max(1, remaining // 10)),
                remaining,
            )

        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)

        safe_next_batch_idx = epoch_start_batch
        pbar = tqdm(
            enumerate(train_loader, start=epoch_start_batch),
            total=full_train_batches,
            initial=epoch_start_batch,
            desc=f"Train {epoch + 1}/{cfg.epochs}",
        )

        try:
            for batch_idx, batch in pbar:
                weak = move_inputs(batch["weak"], device)
                targets = batch["targets"].to(device, non_blocking=True).float()
                domains = batch["domains"].to(device, non_blocking=True)

                use_cons = epoch >= cfg.consistency_start_epoch and cfg.w_cons > 0
                strong = move_inputs(batch["strong"], device) if use_cons else None

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=(cfg.amp and device.type == "cuda"),
                ):
                    weak_out = model(weak)
                    strong_out = model(strong) if use_cons else None
                    loss, parts = total_vapor_loss(
                        cfg,
                        weak_out,
                        strong_out,
                        targets,
                        domains,
                        pos_prior,
                        epoch,
                    )
                    loss = loss / cfg.accumulation_steps

                scaler.scale(loss).backward()
                optimizer_boundary = (batch_idx + 1) % cfg.accumulation_steps == 0

                if optimizer_boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.grad_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    # Safe checkpoint boundary: no unsaved accumulated gradients.
                    safe_next_batch_idx = batch_idx + 1

                raw_loss = float(parts["loss"].detach())
                running += raw_loss * targets.size(0)
                n_seen += targets.size(0)
                pbar.set_postfix(
                    loss=f"{raw_loss:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    resume=f"{epoch_start_batch}",
                )

                if (
                    optimizer_boundary
                    and global_step > 0
                    and global_step % cfg.log_every_steps == 0
                ):
                    row = {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "global_step": global_step,
                        "loss": raw_loss,
                        **{
                            k: float(v.detach())
                            for k, v in parts.items()
                            if k != "loss"
                        },
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    append_csv(res_dir / "step_log.csv", row, step_fields)

                if (
                    cfg.keep_step_checkpoint
                    and optimizer_boundary
                    and global_step > 0
                    and global_step % cfg.save_every_steps == 0
                ):
                    save_checkpoint(
                        resume_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch=epoch,
                        next_batch_idx=safe_next_batch_idx,
                        global_step=global_step,
                        best_score=best_score,
                        cfg=cfg,
                        thresholds=thresholds,
                        calibration=calibration,
                        data_state=_checkpoint_data_state(train_ds, batch_sampler, cfg),
                    )

        except KeyboardInterrupt:
            print(
                f"\n[INTERRUPTED] Saving emergency checkpoint at epoch={epoch}, "
                f"next_batch={safe_next_batch_idx}/{full_train_batches}, "
                f"global_step={global_step}"
            )
            save_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                next_batch_idx=safe_next_batch_idx,
                global_step=global_step,
                best_score=best_score,
                cfg=cfg,
                thresholds=thresholds,
                calibration=calibration,
                data_state=_checkpoint_data_state(train_ds, batch_sampler, cfg),
            )
            print("[INTERRUPTED] Checkpoint saved. Re-run train_colab.py to continue directly from it.")
            raise
        except Exception:
            # Best-effort emergency save for recoverable Python exceptions.
            try:
                print(
                    f"\n[ERROR] Saving emergency checkpoint at epoch={epoch}, "
                    f"next_batch={safe_next_batch_idx}/{full_train_batches}, "
                    f"global_step={global_step}"
                )
                save_checkpoint(
                    resume_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch,
                    next_batch_idx=safe_next_batch_idx,
                    global_step=global_step,
                    best_score=best_score,
                    cfg=cfg,
                    thresholds=thresholds,
                    calibration=calibration,
                    data_state=_checkpoint_data_state(train_ds, batch_sampler, cfg),
                )
            except Exception as save_exc:
                print(f"[WARN] Emergency checkpoint save also failed: {save_exc}")
            raise

        resume_batch = 0

        val_logits, val_targets, val_probs = evaluate(
            model,
            val_loader,
            device,
            None,
            desc=f"Eval {epoch + 1}/{cfg.epochs}",
        )
        thresholds = tune_attribute_thresholds(val_targets, val_probs)
        metrics = evaluate_probs(val_targets, val_probs, thresholds)
        is_best = metrics["challenge_score"] >= best_score
        best_score = max(best_score, metrics["challenge_score"])

        row = {
            "epoch": epoch + 1,
            "resumed_from_batch": epoch_start_batch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running / max(n_seen, 1),
            **metrics,
            "best_score": best_score,
            "seconds": time.time() - t0,
        }
        append_csv(res_dir / "training_log.csv", row, epoch_fields)
        print(row)

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch + 1,
            next_batch_idx=0,
            global_step=global_step,
            best_score=best_score,
            cfg=cfg,
            thresholds=thresholds,
            calibration=calibration,
            data_state=_checkpoint_data_state(train_ds, batch_sampler, cfg),
        )
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch + 1,
                next_batch_idx=0,
                global_step=global_step,
                best_score=best_score,
                cfg=cfg,
                thresholds=thresholds,
                calibration=calibration,
                data_state=_checkpoint_data_state(train_ds, batch_sampler, cfg),
            )
        if resume_path.exists():
            resume_path.unlink()

    # Fit post-hoc calibration using best development checkpoint.
    if best_path.exists():
        load_checkpoint(best_path, model, device="cpu", restore_rng=False)
    raw_logits, y, _ = evaluate(model, val_loader, device, None, desc="Calibration pass")
    calibration = fit_vector_scaling(raw_logits.to(device), y.to(device))
    calibrated = apply_calibration(raw_logits.to(device), calibration).cpu()
    probs = torch.sigmoid(calibrated)
    thresholds = tune_attribute_thresholds(y, probs)
    final_metrics = evaluate_probs(y, probs, thresholds)
    print("Final calibrated validation:", final_metrics)

    save_checkpoint(
        ckpt_dir / "best_calibrated.pt",
        model,
        None,
        None,
        None,
        epoch=cfg.epochs,
        next_batch_idx=0,
        global_step=global_step,
        best_score=final_metrics["challenge_score"],
        cfg=cfg,
        thresholds=thresholds,
        calibration=calibration,
        data_state=_checkpoint_data_state(train_ds, batch_sampler, cfg),
    )
    pd.DataFrame({
        "attribute": ATTRIBUTE_NAMES,
        "threshold": thresholds,
    }).to_csv(res_dir / "thresholds.csv", index=False)

    return {
        "model": model,
        "processor": processor,
        "calibration": calibration,
        "thresholds": thresholds,
        "metrics": final_metrics,
        "best_checkpoint": str(ckpt_dir / "best_calibrated.pt"),
    }
