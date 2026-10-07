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
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument(
        "--output-name",
        type=str,
        default="Submission_PROTOTYPE_AUTOENSEMBLE_TTA.zip",
    )
    p.add_argument("--extra-logit-bias", type=float, default=0.0)
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


def checkpoint_path(root, seed):
    return root / f"prototype_convnext_tiny_seed{seed}_best.pt"


def main():
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    calibration = torch.load(
        args.calibration,
        map_location="cpu",
        weights_only=False,
    )

    members_meta = list(calibration["members"])
    weights = torch.as_tensor(
        calibration["weights"],
        dtype=torch.float32,
    ).reshape(-1)

    threshold_logits = torch.as_tensor(
        calibration["threshold_logits"],
        dtype=torch.float32,
    ).reshape(-1)

    if len(members_meta) != weights.numel():
        raise RuntimeError("Calibration member/weight count mismatch")

    if threshold_logits.numel() != 40:
        raise RuntimeError(
            f"Expected 40 threshold logits, got {threshold_logits.numel()}"
        )

    threshold_logits = threshold_logits + float(args.extra_logit_bias)

    packaged_members = []

    for meta, weight in zip(
        members_meta,
        weights.tolist(),
    ):
        seed = int(meta["seed"])
        p = checkpoint_path(args.checkpoint_dir, seed)

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
            raise RuntimeError(f"{p} has no model state")

        packaged_members.append({
            "seed": seed,
            "weight": float(weight),
            "prototype_config": dict(
                ck.get(
                    "prototype_config",
                    meta.get("prototype_config", {}),
                )
            ),
            "state_dict": half_state_dict(state),
        })

    out = args.result_dir / "_submission_prototype_autoensemble_build"

    if out.exists():
        shutil.rmtree(out)

    (out / "assets").mkdir(parents=True, exist_ok=True)

    package = {
        "format": "upar-hybrid-prototype-autoensemble",
        "members": packaged_members,
        "threshold_logits": threshold_logits,
        "attribute_names": calibration.get(
            "attribute_names",
            ATTRIBUTE_NAMES,
        ),
        "image_height": int(calibration.get("image_height", 288)),
        "image_width": int(calibration.get("image_width", 144)),
        "tta": str(calibration.get("tta", "flip")),
        "calibration_metrics": calibration.get("metrics", {}),
        "calibration_objective": float(
            calibration.get("objective", 0.0)
        ),
        "calibration_robust": float(
            calibration.get("robust", 0.0)
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
                "seed": x["seed"],
                "weight": x["weight"],
                "prototype_config": x["prototype_config"],
            }
            for x in packaged_members
        ],
        "tta": package["tta"],
        "ensemble_space": "raw hybrid logits",
        "calibration": "single post-ensemble threshold vector",
        "extra_logit_bias": float(args.extra_logit_bias),
    }

    (out / "assets" / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    run_py = 'from pathlib import Path\nimport math\nimport os\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom PIL import Image\nfrom torchvision import transforms\nfrom torchvision.models import convnext_tiny\n\n\nROOT = Path(__file__).resolve().parent\n\nPKG = torch.load(\n    ROOT / "assets" / "model.pt",\n    map_location="cpu",\n    weights_only=False,\n)\n\nDEVICE = torch.device(\n    "cuda" if torch.cuda.is_available() else "cpu"\n)\n\nH = int(PKG.get("image_height", 288))\nW = int(PKG.get("image_width", 144))\nBATCH_SIZE = int(\n    os.environ.get(\n        "UPAR_BATCH_SIZE",\n        "40",\n    )\n)\nATTRS = list(PKG["attribute_names"])\nTTA = str(PKG.get("tta", "flip"))\n\nTFM = transforms.Compose([\n    transforms.Resize(\n        (H, W),\n        interpolation=transforms.InterpolationMode.BICUBIC,\n    ),\n    transforms.ToTensor(),\n    transforms.Normalize(\n        [0.485, 0.456, 0.406],\n        [0.229, 0.224, 0.225],\n    ),\n])\n\n\ndef _logit(p):\n    p = min(max(float(p), 1e-5), 1.0 - 1e-5)\n    return math.log(p / (1.0 - p))\n\n\nclass HybridPrototypePAR(nn.Module):\n    def __init__(\n        self,\n        num_attributes=40,\n        prototype_dim=256,\n        prototype_temperature=0.20,\n        prototype_gate_init=0.35,\n        prototype_ema_momentum=0.95,\n        prototype_ema_mix=0.25,\n    ):\n        super().__init__()\n\n        self.num_attributes = int(num_attributes)\n        self.prototype_dim = int(prototype_dim)\n        self.prototype_temperature = float(prototype_temperature)\n        self.prototype_ema_momentum = float(prototype_ema_momentum)\n        self.prototype_ema_mix = float(prototype_ema_mix)\n\n        base = convnext_tiny(weights=None)\n        self.backbone = base\n        self.backbone_dim = int(base.classifier[2].in_features)\n        self.backbone.classifier[2] = nn.Identity()\n\n        self.global_dropout = nn.Dropout(0.22)\n        self.linear_head = nn.Linear(\n            self.backbone_dim,\n            self.num_attributes,\n        )\n\n        self.token_norm = nn.LayerNorm(self.backbone_dim)\n        self.token_proj = nn.Linear(\n            self.backbone_dim,\n            self.prototype_dim,\n            bias=False,\n        )\n\n        self.attribute_queries = nn.Parameter(\n            torch.empty(\n                self.num_attributes,\n                self.prototype_dim,\n            )\n        )\n        nn.init.trunc_normal_(\n            self.attribute_queries,\n            std=0.02,\n        )\n\n        self.attribute_norm = nn.LayerNorm(\n            self.prototype_dim\n        )\n\n        self.prototype_pos = nn.Parameter(\n            torch.empty(\n                self.num_attributes,\n                self.prototype_dim,\n            )\n        )\n        self.prototype_neg = nn.Parameter(\n            torch.empty(\n                self.num_attributes,\n                self.prototype_dim,\n            )\n        )\n\n        nn.init.trunc_normal_(\n            self.prototype_pos,\n            std=0.02,\n        )\n        nn.init.trunc_normal_(\n            self.prototype_neg,\n            std=0.02,\n        )\n\n        with torch.no_grad():\n            pos0 = F.normalize(\n                self.prototype_pos.detach().clone(),\n                dim=-1,\n            )\n            neg0 = F.normalize(\n                self.prototype_neg.detach().clone(),\n                dim=-1,\n            )\n\n        self.register_buffer(\n            "prototype_pos_ema",\n            pos0,\n        )\n        self.register_buffer(\n            "prototype_neg_ema",\n            neg0,\n        )\n        self.register_buffer(\n            "prototype_pos_updates",\n            torch.zeros(\n                self.num_attributes,\n                dtype=torch.long,\n            ),\n        )\n        self.register_buffer(\n            "prototype_neg_updates",\n            torch.zeros(\n                self.num_attributes,\n                dtype=torch.long,\n            ),\n        )\n\n        self.prototype_gate_logits = nn.Parameter(\n            torch.full(\n                (self.num_attributes,),\n                _logit(prototype_gate_init),\n                dtype=torch.float32,\n            )\n        )\n\n    def effective_prototypes(self):\n        learned_pos = F.normalize(\n            self.prototype_pos,\n            dim=-1,\n        )\n        learned_neg = F.normalize(\n            self.prototype_neg,\n            dim=-1,\n        )\n        ema_pos = F.normalize(\n            self.prototype_pos_ema,\n            dim=-1,\n        )\n        ema_neg = F.normalize(\n            self.prototype_neg_ema,\n            dim=-1,\n        )\n\n        mix = float(self.prototype_ema_mix)\n\n        pos = F.normalize(\n            (1.0 - mix) * learned_pos\n            + mix * ema_pos,\n            dim=-1,\n        )\n        neg = F.normalize(\n            (1.0 - mix) * learned_neg\n            + mix * ema_neg,\n            dim=-1,\n        )\n\n        return pos, neg\n\n    def forward(self, x):\n        fmap = self.backbone.features(x)\n\n        pooled = self.backbone.avgpool(fmap)\n        pooled = self.backbone.classifier[0](pooled)\n        global_feature = pooled.flatten(1)\n\n        linear_logits = self.linear_head(\n            self.global_dropout(global_feature)\n        )\n\n        tokens = fmap.permute(\n            0,\n            2,\n            3,\n            1,\n        ).contiguous()\n\n        tokens = self.token_norm(tokens)\n        tokens = self.token_proj(tokens)\n        tokens = tokens.flatten(1, 2)\n\n        q = F.normalize(\n            self.attribute_queries,\n            dim=-1,\n        )\n        k = F.normalize(\n            tokens,\n            dim=-1,\n        )\n\n        attn_logits = torch.einsum(\n            "bnd,cd->bcn",\n            k,\n            q,\n        ) / 0.10\n\n        attn = torch.softmax(\n            attn_logits,\n            dim=-1,\n        )\n\n        features = torch.einsum(\n            "bcn,bnd->bcd",\n            attn,\n            tokens,\n        )\n\n        features = self.attribute_norm(\n            features\n        )\n\n        features = F.normalize(\n            features,\n            dim=-1,\n        )\n\n        pos, neg = self.effective_prototypes()\n\n        pos_sim = torch.einsum(\n            "bcd,cd->bc",\n            features,\n            pos,\n        )\n        neg_sim = torch.einsum(\n            "bcd,cd->bc",\n            features,\n            neg,\n        )\n\n        proto_logits = (\n            pos_sim - neg_sim\n        ) / max(\n            float(self.prototype_temperature),\n            1e-4,\n        )\n\n        gate = torch.sigmoid(\n            self.prototype_gate_logits\n        ).reshape(1, -1)\n\n        gate = gate.clamp(\n            0.08,\n            0.92,\n        )\n\n        return (\n            (1.0 - gate) * linear_logits\n            + gate * proto_logits\n        )\n\n\nMODELS = []\nWEIGHTS = []\n\nfor item in PKG["members"]:\n    cfg = dict(item.get("prototype_config", {}))\n\n    model = HybridPrototypePAR(\n        **cfg,\n    )\n\n    model.load_state_dict(\n        item["state_dict"],\n        strict=True,\n    )\n\n    model = model.to(\n        DEVICE\n    ).eval()\n\n    MODELS.append(model)\n    WEIGHTS.append(\n        float(item["weight"])\n    )\n\nTHRESHOLD_LOGITS = torch.as_tensor(\n    PKG["threshold_logits"],\n    dtype=torch.float32,\n    device=DEVICE,\n).reshape(1, -1)\n\n\ndef _path(sample):\n    for key in (\n        "image_path",\n        "path",\n        "image",\n        "# image",\n        "filename",\n    ):\n        if (\n            key in sample\n            and sample[key] is not None\n        ):\n            return str(sample[key])\n\n    raise KeyError(\n        f"No image path in keys={list(sample.keys())}"\n    )\n\n\ndef _requested(sample):\n    req = (\n        sample.get("attribute_names")\n        or sample.get("attributes")\n    )\n    return list(req) if req else ATTRS\n\n\ndef _load(sample):\n    with Image.open(_path(sample)) as image:\n        return TFM(\n            image.convert("RGB")\n        )\n\n\n@torch.inference_mode()\ndef _model_tta_logits(model, x):\n    with torch.autocast(\n        device_type="cuda",\n        dtype=torch.float16,\n        enabled=DEVICE.type == "cuda",\n    ):\n        z = model(x)\n\n        if TTA == "flip":\n            z = 0.5 * (\n                z\n                + model(\n                    torch.flip(\n                        x,\n                        dims=[3],\n                    )\n                )\n            )\n\n    return z.float()\n\n\n@torch.inference_mode()\ndef _predict(x):\n    total = None\n\n    for model, weight in zip(\n        MODELS,\n        WEIGHTS,\n    ):\n        if weight <= 0.0:\n            continue\n\n        z = (\n            _model_tta_logits(\n                model,\n                x,\n            )\n            * float(weight)\n        )\n\n        total = (\n            z\n            if total is None\n            else total + z\n        )\n\n    if total is None:\n        raise RuntimeError(\n            "All ensemble weights are zero"\n        )\n\n    return torch.sigmoid(\n        total - THRESHOLD_LOGITS\n    )\n\n\ndef _reorder(row, requested):\n    if list(requested) == ATTRS:\n        return row\n\n    lut = {\n        name: i\n        for i, name in enumerate(ATTRS)\n    }\n\n    return [\n        row[lut[name]]\n        for name in requested\n    ]\n\n\ndef predict_batch(samples):\n    out = []\n\n    for start in range(\n        0,\n        len(samples),\n        BATCH_SIZE,\n    ):\n        chunk = samples[\n            start:start + BATCH_SIZE\n        ]\n\n        x = torch.stack([\n            _load(s)\n            for s in chunk\n        ]).to(\n            DEVICE,\n            non_blocking=True,\n        )\n\n        probs = (\n            _predict(x)\n            .cpu()\n            .numpy()\n        )\n\n        for sample, row in zip(\n            chunk,\n            probs,\n        ):\n            out.append(\n                _reorder(\n                    row.tolist(),\n                    _requested(sample),\n                )\n            )\n\n    return out\n\n\ndef predict_image(sample):\n    return predict_batch([sample])[0]\n\n\ndef load_model():\n    return None\n'

    (out / "run.py").write_text(
        run_py,
        encoding="utf-8",
    )

    (out / "metadata.yaml").write_text(
        "name: UPAR Hybrid Prototype AutoEnsemble\n"
        "task: pedestrian_attribute_recognition\n"
        "framework: pytorch\n",
        encoding="utf-8",
    )

    (out / "NOTICE.md").write_text(
        "UPAR Track-1 PAR submission using a ConvNeXt hybrid "
        "positive/negative prototype head, validation-driven weighted "
        "ensemble, post-ensemble calibration, and horizontal-flip TTA.\n",
        encoding="utf-8",
    )

    (out / "LICENSE-model.txt").write_text(
        "Participant-trained weights. Torchvision components retain "
        "their respective upstream licenses.\n",
        encoding="utf-8",
    )

    final = args.result_dir / args.output_name

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
    print("SHA256:", sha256(final))
    print(
        "Members:",
        [
            (
                x["seed"],
                round(float(x["weight"]), 4),
            )
            for x in packaged_members
        ],
    )
    print("TTA:", package["tta"])
    print(
        "Calibration Challenge_Avg:",
        package["calibration_metrics"].get("Challenge_Avg"),
    )


if __name__ == "__main__":
    main()
