from __future__ import annotations

from pathlib import Path
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

from fast_student_dg import ATTRIBUTE_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default="/content/UPAR-Challenge-2027")
    p.add_argument("--checkpoint-dir", default="/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints")
    p.add_argument("--result-dir", default="/content/drive/MyDrive/PedestrianAttributeRecognition/Results")
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--io-workers", type=int, default=8)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no-tta", action="store_false", dest="tta")
    return p.parse_args()


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean(root: Path):
    if not root.exists():
        return
    for d in list(root.rglob("__pycache__")):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    for pat in ("*.pyc", "*.pyo"):
        for p in list(root.rglob(pat)):
            p.unlink(missing_ok=True)
    for p in list(root.rglob(".DS_Store")):
        p.unlink(missing_ok=True)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    ckpt_dir = Path(args.checkpoint_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = Path("/content/submission_dg")
    assets_dir = submission_dir / "assets"
    final_zip = result_dir / "Submission_FAST_STUDENT_DG_256_TTA.zip"

    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    checkpoint_paths = [
        ckpt_dir / f"fast_student_dg_seed{seed}_best.pt"
        for seed in seeds
    ]
    checkpoint_paths = [p for p in checkpoint_paths if p.exists()]
    if not checkpoint_paths:
        raise FileNotFoundError(
            "No DG checkpoints found. Expected e.g. fast_student_dg_seed42_best.pt"
        )

    model_payloads = []
    print("DG checkpoints selected:")
    for path in checkpoint_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        state = (
            obj.get("inference_model_state")
            or obj.get("ema_model_state")
            or obj.get("model_state")
        )
        if not isinstance(state, dict):
            raise RuntimeError(f"No model state in {path}")
        thresholds = obj.get("thresholds")
        if thresholds is None:
            thresholds = torch.full((40,), 0.5, dtype=torch.float32)
        thresholds = torch.as_tensor(thresholds, dtype=torch.float32).reshape(-1)
        if thresholds.numel() != 40:
            raise RuntimeError(f"Wrong thresholds in {path}")

        # Strip train-only domain head and MixStyle keys. MixStyle has no params,
        # but restricting to the inference backbone/head makes the package explicit.
        inference_state = {}
        for k, v in state.items():
            if k.startswith("features.") or k.startswith("attr_head."):
                inference_state[k] = v.detach().cpu()

        print(
            f"  {path.name}: seed={obj.get('seed')} "
            f"source={obj.get('official_metrics_calibrated', obj.get('official_metrics_05', {}))}"
        )
        model_payloads.append({
            "seed": int(obj.get("seed", -1)),
            "state": inference_state,
            "thresholds": thresholds.cpu(),
        })

    package = {
        "format": "upar-fast-student-dg-ensemble-v1",
        "architecture": "efficientnet_b0_dg_inference",
        "num_attributes": 40,
        "attribute_names": ATTRIBUTE_NAMES,
        "image_height": 256,
        "image_width": 128,
        "tta_horizontal_flip": bool(args.tta),
        "models": model_payloads,
    }
    model_file = assets_dir / "model.pt"
    torch.save(package, model_file)
    model_sha = sha256_file(model_file)

    config = {
        "format": package["format"],
        "architecture": package["architecture"],
        "num_attributes": 40,
        "attribute_names": ATTRIBUTE_NAMES,
        "image_height": 256,
        "image_width": 128,
        "batch_size": int(args.batch_size),
        "image_io_workers": int(args.io_workers),
        "tta_horizontal_flip": bool(args.tta),
        "ensemble_size": len(model_payloads),
        "model_sha256": model_sha,
        "offline": True,
    }
    (assets_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata = (
        repo_root / "examples" / "task1" / "sample_code_submission" / "metadata.yaml"
    )
    if not metadata.exists():
        candidates = list(repo_root.rglob("metadata.yaml"))
        candidates = [p for p in candidates if "sample_code_submission" in str(p)]
        if len(candidates) != 1:
            raise FileNotFoundError("Could not locate official Task-1 metadata.yaml")
        metadata = candidates[0]
    metadata_bytes = metadata.read_bytes()
    (submission_dir / "metadata.yaml").write_bytes(metadata_bytes)

    (submission_dir / "NOTICE.md").write_text(
        """# NOTICE\n\nUPAR 2027 domain-generalized Fast Student ensemble.\nInference is offline and uses only torch, torchvision, NumPy and Pillow from the challenge image.\n""",
        encoding="utf-8",
    )
    (submission_dir / "LICENSE-model.txt").write_text(
        """MODEL PACKAGE NOTICE\n\nassets/model.pt contains participant-trained model weights. Third-party components remain under their upstream licenses.\n""",
        encoding="utf-8",
    )

    run_py = r'''from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODEL_FILE = ASSETS / "model.pt"
CONFIG_FILE = ASSETS / "config.json"
with CONFIG_FILE.open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)

BATCH_SIZE = int(CONFIG.get("batch_size", 64))
IO_WORKERS = max(1, int(CONFIG.get("image_io_workers", 8)))
_MODELS = []
_THRESHOLDS = []
_DEVICE = None
_TRANSFORM = None
_ATTRIBUTE_NAMES = None
_ATTRIBUTE_INDEX = None
_POOL = None
_START = None
_COUNT = 0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class InferenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.attr_head = nn.Linear(base.classifier[1].in_features, 40)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.attr_head(x)


def _safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model() -> None:
    global _MODELS, _THRESHOLDS, _DEVICE, _TRANSFORM
    global _ATTRIBUTE_NAMES, _ATTRIBUTE_INDEX, _POOL, _START
    if _MODELS:
        return None

    if CONFIG.get("model_sha256") and _sha256(MODEL_FILE) != CONFIG["model_sha256"]:
        raise RuntimeError("assets/model.pt SHA256 mismatch")

    package = _safe_load(MODEL_FILE)
    _ATTRIBUTE_NAMES = list(package["attribute_names"])
    if len(_ATTRIBUTE_NAMES) != 40:
        raise RuntimeError("Expected 40 attributes")
    _ATTRIBUTE_INDEX = {name: i for i, name in enumerate(_ATTRIBUTE_NAMES)}

    # Codabench currently exposed CPU in observed Track-1 logs; keep CUDA support
    # if a future worker exposes a usable GPU.
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _DEVICE.type == "cuda":
        try:
            torch.empty(1, device=_DEVICE)
        except Exception:
            _DEVICE = torch.device("cpu")

    for item in package["models"]:
        model = InferenceNet()
        model.load_state_dict(item["state"], strict=True)
        model = model.to(_DEVICE).eval().to(memory_format=torch.channels_last)
        if _DEVICE.type == "cuda":
            model = model.half()
        _MODELS.append(model)
        _THRESHOLDS.append(torch.as_tensor(item["thresholds"], dtype=torch.float32, device=_DEVICE))

    h = int(package.get("image_height", 256))
    w = int(package.get("image_width", 128))
    _TRANSFORM = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if _DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        cpu_count = max(1, os.cpu_count() or 1)
        try:
            torch.set_num_threads(min(8, cpu_count))
            torch.set_num_interop_threads(1)
            torch.backends.mkldnn.enabled = True
        except Exception:
            pass

    _POOL = ThreadPoolExecutor(max_workers=IO_WORKERS) if IO_WORKERS > 1 else None
    _START = time.time()
    print(
        f"[UPAR-DG] Python={sys.version.split()[0]} Torch={torch.__version__} "
        f"device={_DEVICE} ensemble={len(_MODELS)} TTA={bool(package.get('tta_horizontal_flip', True))} "
        f"resolution={h}x{w} BATCH_SIZE={BATCH_SIZE}",
        flush=True,
    )
    return None


def _load_image(path):
    with Image.open(path) as im:
        image = im.convert("RGB").copy()
    return _TRANSFORM(image)


def _load_batch(samples):
    paths = []
    for sample in samples:
        path = sample.get("image_path") or sample.get("image")
        if not path:
            raise ValueError("Sample has neither image_path nor image")
        paths.append(path)
    if _POOL is None or len(paths) <= 1:
        tensors = [_load_image(p) for p in paths]
    else:
        tensors = list(_POOL.map(_load_image, paths))
    return torch.stack(tensors, dim=0)


def _calibrated_probs(logits, thresholds):
    eps = 1e-5
    t = thresholds.clamp(eps, 1.0 - eps)
    shift = torch.log(t / (1.0 - t))
    return torch.sigmoid(logits.float() - shift)


@torch.inference_mode()
def _predict_tensor(x):
    x = x.to(
        _DEVICE,
        dtype=torch.float16 if _DEVICE.type == "cuda" else torch.float32,
        non_blocking=(_DEVICE.type == "cuda"),
    ).contiguous(memory_format=torch.channels_last)

    use_tta = bool(CONFIG.get("tta_horizontal_flip", True))
    if use_tta:
        x2 = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
    else:
        x2 = x

    ensemble = []
    n = x.shape[0]
    for model, thresholds in zip(_MODELS, _THRESHOLDS):
        logits = model(x2)
        if use_tta:
            p1 = _calibrated_probs(logits[:n], thresholds)
            p2 = _calibrated_probs(logits[n:], thresholds)
            probs = 0.5 * (p1 + p2)
        else:
            probs = _calibrated_probs(logits, thresholds)
        ensemble.append(probs)
    probs = torch.stack(ensemble, dim=0).mean(dim=0)
    return probs.cpu().numpy().astype(np.float32, copy=False)


def _infer(samples):
    batch = _load_batch(samples)
    try:
        return _predict_tensor(batch)
    except RuntimeError as exc:
        msg = str(exc).lower()
        oom = "out of memory" in msg or "cannot allocate memory" in msg or "not enough memory" in msg
        if oom and len(samples) > 1:
            if _DEVICE.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            mid = len(samples) // 2
            return np.concatenate([_infer(samples[:mid]), _infer(samples[mid:])], axis=0)
        raise


def _reorder(row, requested):
    if requested is None:
        requested = _ATTRIBUTE_NAMES
    if len(requested) != 40:
        raise ValueError("attribute_names must contain 40 names")
    try:
        order = [_ATTRIBUTE_INDEX[name] for name in requested]
    except KeyError as exc:
        raise ValueError(f"Unknown attribute: {exc}") from exc
    row = np.asarray(row, dtype=np.float32)[order]
    if row.shape != (40,) or not np.isfinite(row).all():
        raise ValueError("Invalid prediction row")
    return np.clip(row, 0.0, 1.0).tolist()


def predict_batch(samples: list[dict]) -> list[list[float]]:
    global _COUNT
    load_model()
    if not isinstance(samples, list):
        samples = list(samples)
    if not samples:
        return []
    probs = _infer(samples)
    if probs.shape != (len(samples), 40):
        raise RuntimeError(f"Wrong prediction shape: {probs.shape}")
    out = [_reorder(row, sample.get("attribute_names")) for sample, row in zip(samples, probs)]
    _COUNT += len(samples)
    if _COUNT <= BATCH_SIZE or _COUNT % (BATCH_SIZE * 20) == 0:
        elapsed = max(time.time() - _START, 1e-6)
        print(f"[UPAR-DG] images={_COUNT} avg_img_per_sec={_COUNT/elapsed:.2f}", flush=True)
    return out


def predict_image(sample: dict) -> list[float]:
    return predict_batch([sample])[0]
'''
    (submission_dir / "run.py").write_text(run_py.lstrip(), encoding="utf-8")
    ast.parse((submission_dir / "run.py").read_text(encoding="utf-8"))

    expected = {
        "run.py", "metadata.yaml", "NOTICE.md", "LICENSE-model.txt",
        "assets/model.pt", "assets/config.json",
    }
    clean(submission_dir)
    actual = {
        p.relative_to(submission_dir).as_posix()
        for p in submission_dir.rglob("*") if p.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"Wrong submission structure: {sorted(actual)}")

    # Local smoke test if validation data exists.
    val_csv = repo_root / "data" / "annotations" / "task1" / "val" / "gt.csv"
    if val_csv.exists():
        with val_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        image_col = next((c for c in ("# image", "image", "image_path", "path", "filename") if c in fields), fields[0])

        # Resolve image paths using the same practical search strategy.
        data_root = repo_root / "data"
        index = None
        def resolve(value):
            nonlocal index
            raw = Path(str(value))
            for p in (raw, data_root / raw, repo_root / raw):
                if p.exists():
                    return p
            if index is None:
                index = {}
                for p in data_root.rglob("*"):
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        index.setdefault(p.name, p)
            if raw.name not in index:
                raise FileNotFoundError(value)
            return index[raw.name]

        smoke_n = min(256, len(rows))
        samples = []
        for i in range(smoke_n):
            p = resolve(rows[i][image_col])
            samples.append({
                "index": i,
                "image": rows[i][image_col],
                "image_path": str(p),
                "attribute_names": ATTRIBUTE_NAMES,
            })

        spec = importlib.util.spec_from_file_location("upar_dg_runtime", submission_dir / "run.py")
        runtime = importlib.util.module_from_spec(spec)
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(runtime)
        finally:
            sys.dont_write_bytecode = old
        clean(submission_dir)
        runtime.load_model()
        runtime.predict_batch(samples[: min(16, smoke_n)])
        if getattr(runtime, "_DEVICE", None) is not None and runtime._DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = []
        for start in range(0, smoke_n, runtime.BATCH_SIZE):
            outputs.extend(runtime.predict_batch(samples[start:start + runtime.BATCH_SIZE]))
        if runtime._DEVICE.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        arr = np.asarray(outputs, dtype=np.float32)
        if arr.shape != (smoke_n, 40) or not np.isfinite(arr).all():
            raise RuntimeError("Smoke test failed")
        rate = smoke_n / max(sec, 1e-6)
        print(f"Smoke: {rate:.2f} img/s; projected hidden test: {29248/rate/60:.2f} min on this runtime")

    clean(submission_dir)
    if final_zip.exists():
        final_zip.unlink()
    with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        for p in sorted(submission_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(submission_dir).as_posix()
            if rel not in expected:
                raise RuntimeError(f"Unexpected file: {rel}")
            zf.write(p, arcname=rel)

    with zipfile.ZipFile(final_zip, "r") as zf:
        names = {n for n in zf.namelist() if not n.endswith("/")}
        if names != expected:
            raise RuntimeError(f"Final ZIP structure wrong: {sorted(names)}")
        if zf.read("metadata.yaml") != metadata_bytes:
            raise RuntimeError("metadata.yaml changed")

    print("\nDG submission ready")
    print("Ensemble size:", len(model_payloads))
    print("Resolution: 256x128")
    print("Horizontal flip TTA:", bool(args.tta))
    print("ZIP:", final_zip)
    print("ZIP MiB:", round(final_zip.stat().st_size / 1024**2, 2))


if __name__ == "__main__":
    main()
