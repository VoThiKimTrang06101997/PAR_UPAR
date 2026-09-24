from pathlib import Path
import os
import random
import numpy as np
import torch


def torch_load_compat(path, map_location="cpu"):
    """Load across PyTorch versions where weights_only default changed."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def compact_state_dict(model):
    """Avoid re-saving the full frozen SigLIP2 backbone."""
    req = {n for n, p in model.named_parameters() if p.requires_grad}
    state = model.state_dict()
    keep = {}
    for k, v in state.items():
        # Keep all VAPOR-PAR heads/buffers; for VLM keep only trainable backbone params.
        if not k.startswith("vlm.") or k in req:
            keep[k] = v.detach().cpu()
    return keep


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as exc:
        print(f"[WARN] Could not fully restore RNG state: {exc}")


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    next_batch_idx,
    global_step,
    best_score,
    cfg,
    thresholds=None,
    calibration=None,
    data_state=None,
):
    payload = {
        "checkpoint_version": 3,
        "model": compact_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": int(epoch),
        "next_batch_idx": int(next_batch_idx),
        "global_step": int(global_step),
        "best_score": float(best_score),
        "cfg": cfg.to_dict(),
        "thresholds": thresholds,
        "calibration": calibration,
        "data_state": data_state or {},
        "rng_state": _capture_rng_state(),
    }
    atomic_torch_save(payload, path)


def load_checkpoint_payload(path, device="cpu"):
    return torch_load_compat(path, map_location=device)


def apply_checkpoint_payload(
    ckpt,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    restore_rng=True,
):
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    # Frozen VLM weights are intentionally absent from compact training checkpoints.
    missing = [m for m in missing if not m.startswith("vlm.")]
    if missing:
        print("[WARN] Missing non-VLM keys:", missing[:20])
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:20])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        restore_rng_state(ckpt.get("rng_state"))
    return ckpt


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu",
    restore_rng=True,
):
    ckpt = load_checkpoint_payload(path, device=device)
    return apply_checkpoint_payload(
        ckpt,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_rng=restore_rng,
    )
