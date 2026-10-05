from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--official-root",
        type=Path,
        default=Path("/content/UPAR-Challenge-2027"),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints"),
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Results"),
    )
    p.add_argument(
        "--members",
        default="convnext_tiny:42,swin_v2_t:123",
        help="comma-separated backbone:seed pairs",
    )
    p.add_argument(
        "--tta",
        choices=["none", "flip", "flip-scale"],
        default="flip",
    )
    p.add_argument(
        "--calibration-bias",
        type=float,
        default=0.12,
        help="positive logit bias raises the effective decision threshold",
    )
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument(
        "--output-name",
        default="Submission_STRONG_TTA.zip",
    )
    return p.parse_args()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def half_state_dict(state):
    out = {}
    for k, v in state.items():
        t = v.detach().cpu()
        if torch.is_floating_point(t):
            t = t.half()
        out[k] = t
    return out


def parse_members(spec: str):
    items = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        backbone, seed = chunk.split(":")
        items.append((backbone.strip(), int(seed)))
    if not items:
        raise ValueError("No ensemble members specified")
    return items


def main():
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    out = Path("/content/submission_strong_tta")
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)

    members = []
    attrs = None
    image_height = 288
    image_width = 144

    for backbone, seed in parse_members(args.members):
        p = args.checkpoint_dir / f"strong_{backbone}_seed{seed}_best.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"Required ensemble checkpoint missing: {p}\n"
                "Train/resume all requested members before export."
            )

        ck = torch.load(p, map_location="cpu", weights_only=False)
        state = ck.get("inference_model_state")
        thresholds = ck.get("thresholds")

        if state is None:
            raise RuntimeError(
                f"{p} has no inference_model_state; it is not a finalized strong-v2 checkpoint."
            )
        if thresholds is None:
            raise RuntimeError(
                f"{p} has no calibrated thresholds. Resume train_strong_par.py so calibration runs."
            )

        th = torch.as_tensor(thresholds, dtype=torch.float32).reshape(-1)
        if th.numel() != 40:
            raise RuntimeError(f"{p}: expected 40 thresholds, got {th.numel()}")

        attrs = ck.get("attribute_names") or attrs
        image_height = int(ck.get("image_height", image_height))
        image_width = int(ck.get("image_width", image_width))

        members.append({
            "backbone": backbone,
            "seed": seed,
            "state_dict": half_state_dict(state),
            "thresholds": th,
            "selection": float(ck.get("selection", 0.0)),
            "selection_calibrated_robust": float(
                ck.get("selection_calibrated_robust", 0.0)
            ),
            "metrics": ck.get("metrics_calibrated", {}),
            "source": ck.get("source", ""),
        })

    package = {
        "format": "upar-strong-tta-v2",
        "members": members,
        "image_height": image_height,
        "image_width": image_width,
        "tta": args.tta,
        "calibration_bias": float(args.calibration_bias),
        "attribute_names": attrs,
    }
    torch.save(package, out / "assets" / "model.pt")

    config = {
        "ensemble": [
            {"backbone": m["backbone"], "seed": m["seed"]}
            for m in members
        ],
        "storage_dtype": "float16",
        "runtime_dtype": "autocast-fp16-on-cuda",
        "image_height": image_height,
        "image_width": image_width,
        "tta": args.tta,
        "calibration_bias": float(args.calibration_bias),
        "batch_size": int(args.batch_size),
        "ensemble_space": "threshold-centered logits",
    }
    (out / "assets" / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    sample_meta = (
        args.official_root
        / "examples"
        / "task1"
        / "sample_code_submission"
        / "metadata.yaml"
    )
    if not sample_meta.exists():
        raise FileNotFoundError(sample_meta)
    shutil.copy2(sample_meta, out / "metadata.yaml")

    (out / "NOTICE.md").write_text(
        "UPAR 2027 Track-1 PAR. ConvNeXt-Tiny + Swin-V2-T ensemble, "
        "top-k weight soup, precision-aware source calibration and TTA. "
        "Offline inference only.\n",
        encoding="utf-8",
    )
    (out / "LICENSE-model.txt").write_text(
        "Participant-trained weights. Torchvision backbone components retain "
        "their upstream licenses.\n",
        encoding="utf-8",
    )

    run_py = r'''from __future__ import annotations

from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny, swin_v2_t


ROOT = Path(__file__).resolve().parent
PKG = torch.load(ROOT / "assets" / "model.pt", map_location="cpu", weights_only=False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H = int(PKG.get("image_height", 288))
W = int(PKG.get("image_width", 144))
BATCH_SIZE = int(os.environ.get("UPAR_BATCH_SIZE", "40"))
ATTRS = PKG["attribute_names"]
TTA = str(PKG.get("tta", "flip"))
CAL_BIAS = float(PKG.get("calibration_bias", 0.0))

TFM = transforms.Compose([
    transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])


class StrongPARModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "convnext_tiny":
            base = convnext_tiny(weights=None)
            d = int(base.classifier[2].in_features)
            base.classifier[2] = nn.Sequential(
                nn.Dropout(0.22),
                nn.Linear(d, 40),
            )
        elif backbone == "swin_v2_t":
            base = swin_v2_t(weights=None)
            d = int(base.head.in_features)
            base.head = nn.Sequential(
                nn.Dropout(0.22),
                nn.Linear(d, 40),
            )
        else:
            raise ValueError(backbone)
        self.net = base

    def forward(self, x):
        return self.net(x)


MODELS = []
THRESHOLD_LOGITS = []

for item in PKG["members"]:
    model = StrongPARModel(item["backbone"])
    model.load_state_dict(item["state_dict"], strict=True)
    model = model.to(DEVICE).eval()
    MODELS.append(model)

    th = torch.as_tensor(
        item["thresholds"],
        dtype=torch.float32,
        device=DEVICE,
    ).clamp(1e-5, 1.0 - 1e-5)
    THRESHOLD_LOGITS.append(torch.logit(th))


def _path(sample):
    for k in ("image_path", "path", "image", "# image", "filename"):
        if k in sample and sample[k] is not None:
            return str(sample[k])
    raise KeyError(f"No image path in sample keys={list(sample.keys())}")


def _requested(sample):
    req = sample.get("attribute_names") or sample.get("attributes")
    return req if req else ATTRS


def _load(sample):
    with Image.open(_path(sample)) as im:
        return TFM(im.convert("RGB"))


def _scale_center_view(x):
    hh = H + 24
    ww = W + 12
    big = F.interpolate(
        x,
        size=(hh, ww),
        mode="bicubic",
        align_corners=False,
    )
    top = (hh - H) // 2
    left = (ww - W) // 2
    return big[:, :, top:top+H, left:left+W]


@torch.inference_mode()
def _model_tta_logits(model, x):
    views = [x]

    if TTA in ("flip", "flip-scale"):
        views.append(torch.flip(x, dims=[3]))

    if TTA == "flip-scale":
        xs = _scale_center_view(x)
        views.extend([xs, torch.flip(xs, dims=[3])])

    total = None
    for view in views:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda",
        ):
            z = model(view)
        z = z.float()
        total = z if total is None else total + z

    return total / float(len(views))


@torch.inference_mode()
def _predict(x):
    total = None
    for model, threshold_logit in zip(MODELS, THRESHOLD_LOGITS):
        z = _model_tta_logits(model, x)
        z = z - threshold_logit.reshape(1, -1) - CAL_BIAS
        total = z if total is None else total + z

    z = total / float(len(MODELS))
    return torch.sigmoid(z)


def _reorder(row, requested):
    if list(requested) == list(ATTRS):
        return row
    lut = {a: i for i, a in enumerate(ATTRS)}
    return [row[lut[a]] for a in requested]


def predict_batch(samples: list[dict]) -> list[list[float]]:
    out = []
    for st in range(0, len(samples), BATCH_SIZE):
        chunk = samples[st:st+BATCH_SIZE]
        x = torch.stack([_load(s) for s in chunk]).to(
            DEVICE,
            non_blocking=True,
        )
        p = _predict(x).cpu().numpy()
        for sample, row in zip(chunk, p):
            out.append(_reorder(row.tolist(), _requested(sample)))
    return out


def predict_image(sample: dict) -> list[float]:
    return predict_batch([sample])[0]


def load_model():
    return None
'''
    (out / "run.py").write_text(run_py, encoding="utf-8")

    final = args.result_dir / args.output_name
    if final.exists():
        final.unlink()

    with zipfile.ZipFile(final, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if (
                p.is_file()
                and "__pycache__" not in p.parts
                and p.suffix not in {".pyc", ".pyo"}
            ):
                z.write(p, p.relative_to(out))

    print("Built:", final)
    print("MiB:", final.stat().st_size / 1024 / 1024)
    print("SHA256:", sha256(final))
    print(
        "Members:",
        [(m["backbone"], m["seed"]) for m in members],
    )
    print("TTA:", args.tta, "| calibration_bias:", args.calibration_bias)


if __name__ == "__main__":
    main()
