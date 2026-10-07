from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch


ATTRIBUTE_NAMES = [
    "Age-Young","Age-Adult","Age-Old","Gender-Female",
    "Hair-Length-Short","Hair-Length-Long","Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black","UpperBody-Color-Blue","UpperBody-Color-Brown",
    "UpperBody-Color-Green","UpperBody-Color-Grey","UpperBody-Color-Orange",
    "UpperBody-Color-Pink","UpperBody-Color-Purple","UpperBody-Color-Red",
    "UpperBody-Color-White","UpperBody-Color-Yellow","UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black","LowerBody-Color-Blue","LowerBody-Color-Brown",
    "LowerBody-Color-Green","LowerBody-Color-Grey","LowerBody-Color-Orange",
    "LowerBody-Color-Pink","LowerBody-Color-Purple","LowerBody-Color-Red",
    "LowerBody-Color-White","LowerBody-Color-Yellow","LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts","LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack","Accessory-Bag","Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun","Accessory-Hat",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="Submission_AUTOENSEMBLE_TTA.zip",
    )
    p.add_argument(
        "--extra-logit-bias",
        type=float,
        default=0.0,
    )
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def half_state_dict(state):
    out = {}
    for key, value in state.items():
        t = value.detach().cpu()
        if torch.is_floating_point(t):
            t = t.half()
        out[key] = t
    return out


def checkpoint_path(root, backbone, seed):
    return (
        root
        / f"strong_{backbone}_seed{seed}_best.pt"
    )


def main():
    args = parse_args()
    args.result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration = torch.load(
        args.calibration,
        map_location="cpu",
        weights_only=False,
    )

    members_meta = calibration["members"]

    weights = torch.as_tensor(
        calibration["weights"],
        dtype=torch.float32,
    ).reshape(-1)

    threshold_logits = torch.as_tensor(
        calibration["threshold_logits"],
        dtype=torch.float32,
    ).reshape(-1)

    if len(members_meta) != weights.numel():
        raise RuntimeError(
            "Calibration member/weight count mismatch"
        )

    if threshold_logits.numel() != 40:
        raise RuntimeError(
            f"Expected 40 thresholds, got {threshold_logits.numel()}"
        )

    threshold_logits = (
        threshold_logits
        + float(args.extra_logit_bias)
    )

    packaged_members = []

    for meta, weight in zip(
        members_meta,
        weights.tolist(),
    ):
        backbone = str(meta["backbone"])
        seed = int(meta["seed"])

        p = checkpoint_path(
            args.checkpoint_dir,
            backbone,
            seed,
        )

        if not p.exists():
            raise FileNotFoundError(p)

        ck = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        state = (
            ck.get("inference_model_state")
            or ck.get("ema_model_state")
            or ck.get("model_state")
        )

        if state is None:
            raise RuntimeError(
                f"{p} has no model state"
            )

        packaged_members.append({
            "backbone": backbone,
            "seed": seed,
            "weight": float(weight),
            "state_dict": half_state_dict(state),
        })

    out = Path(
        "/content/submission_autoensemble"
    )

    if out.exists():
        shutil.rmtree(out)

    (out / "assets").mkdir(
        parents=True,
        exist_ok=True,
    )

    package = {
        "format": "upar-autoensemble",
        "members": packaged_members,
        "threshold_logits": threshold_logits,
        "attribute_names": calibration.get(
            "attribute_names",
            ATTRIBUTE_NAMES,
        ),
        "image_height": int(
            calibration.get(
                "image_height",
                288,
            )
        ),
        "image_width": int(
            calibration.get(
                "image_width",
                144,
            )
        ),
        "tta": str(
            calibration.get(
                "tta",
                "flip",
            )
        ),
        "calibration_metrics": calibration.get(
            "metrics",
            {},
        ),
        "calibration_objective": float(
            calibration.get(
                "objective",
                0.0,
            )
        ),
    }

    torch.save(
        package,
        out / "assets" / "model.pt",
    )

    config = {
        "format": package["format"],
        "members": [
            {
                "backbone": x["backbone"],
                "seed": x["seed"],
                "weight": x["weight"],
            }
            for x in packaged_members
        ],
        "tta": package["tta"],
        "ensemble_space": "raw logits",
        "calibration": (
            "single post-ensemble threshold vector"
        ),
        "extra_logit_bias": float(
            args.extra_logit_bias
        ),
    }

    (out / "assets" / "config.json").write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    run_py = r'''from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny, swin_v2_t


ROOT = Path(__file__).resolve().parent

PKG = torch.load(
    ROOT / "assets" / "model.pt",
    map_location="cpu",
    weights_only=False,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

H = int(PKG.get("image_height", 288))
W = int(PKG.get("image_width", 144))
BATCH_SIZE = int(
    os.environ.get(
        "UPAR_BATCH_SIZE",
        "48",
    )
)
ATTRS = list(PKG["attribute_names"])
TTA = str(PKG.get("tta", "flip"))

TFM = transforms.Compose([
    transforms.Resize(
        (H, W),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


class StrongPARModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        if backbone == "convnext_tiny":
            base = convnext_tiny(weights=None)
            d = int(
                base.classifier[2].in_features
            )
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
WEIGHTS = []

for item in PKG["members"]:
    model = StrongPARModel(
        item["backbone"]
    )

    model.load_state_dict(
        item["state_dict"],
        strict=True,
    )

    model = model.to(
        DEVICE
    ).eval()

    MODELS.append(model)
    WEIGHTS.append(
        float(item["weight"])
    )

THRESHOLD_LOGITS = torch.as_tensor(
    PKG["threshold_logits"],
    dtype=torch.float32,
    device=DEVICE,
).reshape(1, -1)


def _path(sample):
    for key in (
        "image_path",
        "path",
        "image",
        "# image",
        "filename",
    ):
        if (
            key in sample
            and sample[key] is not None
        ):
            return str(sample[key])

    raise KeyError(
        f"No image path in keys={list(sample.keys())}"
    )


def _requested(sample):
    req = (
        sample.get("attribute_names")
        or sample.get("attributes")
    )
    return list(req) if req else ATTRS


def _load(sample):
    with Image.open(_path(sample)) as im:
        return TFM(
            im.convert("RGB")
        )


@torch.inference_mode()
def _model_tta_logits(model, x):
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=DEVICE.type == "cuda",
    ):
        z = model(x)

        if TTA == "flip":
            z = 0.5 * (
                z
                + model(
                    torch.flip(
                        x,
                        dims=[3],
                    )
                )
            )

    return z.float()


@torch.inference_mode()
def _predict(x):
    total = None

    for model, weight in zip(
        MODELS,
        WEIGHTS,
    ):
        if weight <= 0.0:
            continue

        z = (
            _model_tta_logits(
                model,
                x,
            )
            * float(weight)
        )

        total = (
            z
            if total is None
            else total + z
        )

    if total is None:
        raise RuntimeError(
            "All ensemble weights are zero"
        )

    return torch.sigmoid(
        total - THRESHOLD_LOGITS
    )


def _reorder(row, requested):
    if list(requested) == ATTRS:
        return row

    lut = {
        name: i
        for i, name in enumerate(ATTRS)
    }

    return [
        row[lut[name]]
        for name in requested
    ]


def predict_batch(samples):
    out = []

    for start in range(
        0,
        len(samples),
        BATCH_SIZE,
    ):
        chunk = samples[
            start:start + BATCH_SIZE
        ]

        x = torch.stack([
            _load(s)
            for s in chunk
        ]).to(
            DEVICE,
            non_blocking=True,
        )

        probs = (
            _predict(x)
            .cpu()
            .numpy()
        )

        for sample, row in zip(
            chunk,
            probs,
        ):
            out.append(
                _reorder(
                    row.tolist(),
                    _requested(sample),
                )
            )

    return out


def predict_image(sample):
    return predict_batch([sample])[0]


def load_model():
    return None
'''

    (out / "run.py").write_text(
        run_py,
        encoding="utf-8",
    )

    (out / "metadata.yaml").write_text(
        "name: UPAR AutoEnsemble\n"
        "task: pedestrian_attribute_recognition\n"
        "framework: pytorch\n",
        encoding="utf-8",
    )

    (out / "NOTICE.md").write_text(
        "UPAR 2027 Track-1 PAR. "
        "Validation-driven weighted ensemble, "
        "post-ensemble calibration, flip-TTA.\n",
        encoding="utf-8",
    )

    (out / "LICENSE-model.txt").write_text(
        "Participant-trained weights. "
        "Torchvision components retain "
        "their upstream licenses.\n",
        encoding="utf-8",
    )

    final = (
        args.result_dir
        / args.output_name
    )

    if final.exists():
        final.unlink()

    with zipfile.ZipFile(
        final,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:
        for p in out.rglob("*"):
            if (
                p.is_file()
                and "__pycache__" not in p.parts
                and p.suffix not in {".pyc", ".pyo"}
            ):
                z.write(
                    p,
                    p.relative_to(out),
                )

    print("Built:", final)
    print(
        "MiB:",
        final.stat().st_size / 1024 / 1024,
    )
    print(
        "SHA256:",
        sha256(final),
    )
    print(
        "Members:",
        [
            (
                x["backbone"],
                x["seed"],
                round(
                    float(x["weight"]),
                    4,
                ),
            )
            for x in packaged_members
        ],
    )
    print(
        "TTA:",
        package["tta"],
    )
    print(
        "Calibration Challenge_Avg:",
        package[
            "calibration_metrics"
        ].get(
            "Challenge_Avg"
        ),
    )


if __name__ == "__main__":
    main()
