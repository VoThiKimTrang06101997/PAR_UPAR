from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
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

RUNTIME_SOURCE = '"""UPAR Track-1 prototype submission runtime.\n\nRequired API:\n    load_model() -> None\n    predict_image(sample: dict) -> list[float]\n    predict_batch(samples: list[dict]) -> list[list[float]]\n\nThe evaluator provides absolute image paths in sample["image_path"].\n"""\nfrom __future__ import annotations\n\nimport gc\nimport math\nimport os\nfrom pathlib import Path\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom PIL import Image\nfrom torchvision import transforms\nfrom torchvision.models import convnext_tiny\n\n\nROOT = Path(__file__).resolve().parent\nBATCH_SIZE = 64\nMICRO_BATCH_SIZE = max(\n    1,\n    int(os.environ.get("UPAR_MICRO_BATCH_SIZE", "__MICRO_BATCH_SIZE__")),\n)\n\n_MODEL = None\n\n\ndef _safe_torch_load(path, map_location="cpu"):\n    try:\n        return torch.load(\n            path,\n            map_location=map_location,\n            weights_only=False,\n        )\n    except TypeError:\n        return torch.load(\n            path,\n            map_location=map_location,\n        )\n\n\ndef _logit(p):\n    p = min(max(float(p), 1e-5), 1.0 - 1e-5)\n    return math.log(p / (1.0 - p))\n\n\nclass HybridPrototypePAR(nn.Module):\n    def __init__(\n        self,\n        num_attributes=40,\n        prototype_dim=256,\n        prototype_temperature=0.20,\n        prototype_gate_init=0.35,\n        prototype_ema_momentum=0.95,\n        prototype_ema_mix=0.25,\n    ):\n        super().__init__()\n\n        self.num_attributes = int(num_attributes)\n        self.prototype_dim = int(prototype_dim)\n        self.prototype_temperature = float(prototype_temperature)\n        self.prototype_ema_momentum = float(prototype_ema_momentum)\n        self.prototype_ema_mix = float(prototype_ema_mix)\n\n        base = convnext_tiny(weights=None)\n        self.backbone = base\n        self.backbone_dim = int(base.classifier[2].in_features)\n        self.backbone.classifier[2] = nn.Identity()\n\n        self.global_dropout = nn.Dropout(0.22)\n        self.linear_head = nn.Linear(\n            self.backbone_dim,\n            self.num_attributes,\n        )\n\n        self.token_norm = nn.LayerNorm(self.backbone_dim)\n        self.token_proj = nn.Linear(\n            self.backbone_dim,\n            self.prototype_dim,\n            bias=False,\n        )\n\n        self.attribute_queries = nn.Parameter(\n            torch.zeros(\n                self.num_attributes,\n                self.prototype_dim,\n            )\n        )\n\n        self.attribute_norm = nn.LayerNorm(\n            self.prototype_dim\n        )\n\n        self.prototype_pos = nn.Parameter(\n            torch.zeros(\n                self.num_attributes,\n                self.prototype_dim,\n            )\n        )\n        self.prototype_neg = nn.Parameter(\n            torch.zeros(\n                self.num_attributes,\n                self.prototype_dim,\n            )\n        )\n\n        self.register_buffer(\n            "prototype_pos_ema",\n            torch.zeros(\n                self.num_attributes,\n                self.prototype_dim,\n            ),\n        )\n        self.register_buffer(\n            "prototype_neg_ema",\n            torch.zeros(\n                self.num_attributes,\n                self.prototype_dim,\n            ),\n        )\n        self.register_buffer(\n            "prototype_pos_updates",\n            torch.zeros(\n                self.num_attributes,\n                dtype=torch.long,\n            ),\n        )\n        self.register_buffer(\n            "prototype_neg_updates",\n            torch.zeros(\n                self.num_attributes,\n                dtype=torch.long,\n            ),\n        )\n\n        self.prototype_gate_logits = nn.Parameter(\n            torch.full(\n                (self.num_attributes,),\n                _logit(prototype_gate_init),\n                dtype=torch.float32,\n            )\n        )\n\n    def effective_prototypes(self):\n        learned_pos = F.normalize(\n            self.prototype_pos,\n            dim=-1,\n        )\n        learned_neg = F.normalize(\n            self.prototype_neg,\n            dim=-1,\n        )\n        ema_pos = F.normalize(\n            self.prototype_pos_ema,\n            dim=-1,\n        )\n        ema_neg = F.normalize(\n            self.prototype_neg_ema,\n            dim=-1,\n        )\n\n        mix = float(self.prototype_ema_mix)\n\n        pos = F.normalize(\n            (1.0 - mix) * learned_pos\n            + mix * ema_pos,\n            dim=-1,\n        )\n        neg = F.normalize(\n            (1.0 - mix) * learned_neg\n            + mix * ema_neg,\n            dim=-1,\n        )\n        return pos, neg\n\n    def forward(self, x):\n        fmap = self.backbone.features(x)\n\n        pooled = self.backbone.avgpool(fmap)\n        pooled = self.backbone.classifier[0](pooled)\n        global_feature = pooled.flatten(1)\n\n        linear_logits = self.linear_head(\n            self.global_dropout(global_feature)\n        )\n\n        tokens = fmap.permute(\n            0,\n            2,\n            3,\n            1,\n        ).contiguous()\n\n        tokens = self.token_norm(tokens)\n        tokens = self.token_proj(tokens)\n        tokens = tokens.flatten(1, 2)\n\n        q = F.normalize(\n            self.attribute_queries,\n            dim=-1,\n        )\n        k = F.normalize(\n            tokens,\n            dim=-1,\n        )\n\n        attn_logits = torch.einsum(\n            "bnd,cd->bcn",\n            k,\n            q,\n        ) / 0.10\n\n        attn = torch.softmax(\n            attn_logits,\n            dim=-1,\n        )\n\n        features = torch.einsum(\n            "bcn,bnd->bcd",\n            attn,\n            tokens,\n        )\n\n        features = self.attribute_norm(\n            features\n        )\n        features = F.normalize(\n            features,\n            dim=-1,\n        )\n\n        pos, neg = self.effective_prototypes()\n\n        pos_sim = torch.einsum(\n            "bcd,cd->bc",\n            features,\n            pos,\n        )\n        neg_sim = torch.einsum(\n            "bcd,cd->bc",\n            features,\n            neg,\n        )\n\n        proto_logits = (\n            pos_sim - neg_sim\n        ) / max(\n            float(self.prototype_temperature),\n            1e-4,\n        )\n\n        gate = torch.sigmoid(\n            self.prototype_gate_logits\n        ).reshape(1, -1)\n\n        gate = gate.clamp(\n            0.08,\n            0.92,\n        )\n\n        return (\n            (1.0 - gate) * linear_logits\n            + gate * proto_logits\n        )\n\n\ndef _sanitize_config(config, state):\n    config = dict(config or {})\n\n    if "linear_head.weight" in state:\n        config["num_attributes"] = int(\n            state["linear_head.weight"].shape[0]\n        )\n\n    if "attribute_queries" in state:\n        config["prototype_dim"] = int(\n            state["attribute_queries"].shape[1]\n        )\n\n    defaults = {\n        "num_attributes": 40,\n        "prototype_dim": 256,\n        "prototype_temperature": 0.20,\n        "prototype_gate_init": 0.35,\n        "prototype_ema_momentum": 0.95,\n        "prototype_ema_mix": 0.25,\n    }\n\n    return {\n        key: config.get(key, default)\n        for key, default in defaults.items()\n    }\n\n\nclass PrototypeRuntime:\n    def __init__(self, bundle_path):\n        self.device = torch.device(\n            "cuda"\n            if torch.cuda.is_available()\n            else "cpu"\n        )\n\n        bundle = _safe_torch_load(\n            bundle_path,\n            map_location="cpu",\n        )\n\n        self.attribute_names = list(\n            bundle["attribute_names"]\n        )\n\n        if len(self.attribute_names) != 40:\n            raise ValueError(\n                "Submission bundle must contain exactly 40 attributes."\n            )\n\n        self.name_to_index = {\n            name: i\n            for i, name in enumerate(self.attribute_names)\n        }\n\n        self.image_height = int(\n            bundle.get("image_height", 288)\n        )\n        self.image_width = int(\n            bundle.get("image_width", 144)\n        )\n        self.tta = str(\n            bundle.get("tta", "flip")\n        )\n\n        threshold_logits = torch.as_tensor(\n            bundle["threshold_logits"],\n            dtype=torch.float32,\n        ).reshape(-1)\n\n        if threshold_logits.numel() != 40:\n            raise ValueError(\n                "Expected exactly 40 threshold logits."\n            )\n\n        self.threshold_logits = (\n            threshold_logits\n            .reshape(1, -1)\n            .to(self.device)\n        )\n\n        self.transform = transforms.Compose([\n            transforms.Resize(\n                (\n                    self.image_height,\n                    self.image_width,\n                ),\n                interpolation=(\n                    transforms.InterpolationMode.BICUBIC\n                ),\n            ),\n            transforms.ToTensor(),\n            transforms.Normalize(\n                [0.485, 0.456, 0.406],\n                [0.229, 0.224, 0.225],\n            ),\n        ])\n\n        members = list(\n            bundle.get("members", [])\n        )\n\n        if not members:\n            raise ValueError(\n                "Submission bundle contains no ensemble members."\n            )\n\n        self.models = []\n        self.weights = []\n\n        for item in members:\n            weight = float(\n                item.get("weight", 0.0)\n            )\n\n            if weight <= 0.0:\n                item["state_dict"] = None\n                continue\n\n            state = item.get("state_dict")\n\n            if not isinstance(state, dict):\n                raise TypeError(\n                    "Ensemble member has no valid state_dict."\n                )\n\n            cfg = _sanitize_config(\n                item.get(\n                    "prototype_config",\n                    {},\n                ),\n                state,\n            )\n\n            model = HybridPrototypePAR(\n                **cfg,\n            )\n\n            model.load_state_dict(\n                state,\n                strict=True,\n            )\n\n            item["state_dict"] = None\n            del state\n\n            model = model.to(\n                self.device\n            ).eval()\n\n            self.models.append(model)\n            self.weights.append(weight)\n\n        del members\n        del bundle\n        gc.collect()\n\n        if self.device.type == "cuda":\n            torch.cuda.empty_cache()\n\n        if not self.models:\n            raise RuntimeError(\n                "All ensemble members have zero weight."\n            )\n\n        weight_sum = sum(self.weights)\n\n        if (\n            not math.isfinite(weight_sum)\n            or weight_sum <= 0\n        ):\n            raise RuntimeError(\n                "Invalid ensemble weights."\n            )\n\n        self.weights = [\n            float(w / weight_sum)\n            for w in self.weights\n        ]\n\n    def _open_image(self, path):\n        with Image.open(path) as image:\n            return self.transform(\n                image.convert("RGB")\n            )\n\n    def _requested_indices(self, attribute_names):\n        requested = list(attribute_names)\n\n        if len(requested) != 40:\n            raise ValueError(\n                f"Expected 40 attributes, got {len(requested)}"\n            )\n\n        unknown = [\n            x\n            for x in requested\n            if x not in self.name_to_index\n        ]\n\n        if unknown:\n            raise KeyError(\n                "Challenge requested unknown attributes: "\n                + ", ".join(unknown)\n            )\n\n        return [\n            self.name_to_index[x]\n            for x in requested\n        ]\n\n    def _model_logits(self, model, x):\n        if self.device.type == "cuda":\n            with torch.autocast(\n                device_type="cuda",\n                dtype=torch.float16,\n            ):\n                logits = model(x)\n\n                if self.tta == "flip":\n                    logits_flip = model(\n                        torch.flip(\n                            x,\n                            dims=[3],\n                        )\n                    )\n                    logits = 0.5 * (\n                        logits\n                        + logits_flip\n                    )\n        else:\n            logits = model(x)\n\n            if self.tta == "flip":\n                logits_flip = model(\n                    torch.flip(\n                        x,\n                        dims=[3],\n                    )\n                )\n                logits = 0.5 * (\n                    logits\n                    + logits_flip\n                )\n\n        return logits.float()\n\n    @torch.inference_mode()\n    def _predict_tensor(self, x):\n        total = None\n\n        for model, weight in zip(\n            self.models,\n            self.weights,\n        ):\n            logits = (\n                self._model_logits(\n                    model,\n                    x,\n                )\n                * float(weight)\n            )\n\n            total = (\n                logits\n                if total is None\n                else total + logits\n            )\n\n        if total is None:\n            raise RuntimeError(\n                "No active ensemble members."\n            )\n\n        probs = torch.sigmoid(\n            total - self.threshold_logits\n        )\n\n        return torch.nan_to_num(\n            probs.float(),\n            nan=0.5,\n            posinf=1.0,\n            neginf=0.0,\n        ).clamp(0.0, 1.0)\n\n    def predict_samples(\n        self,\n        samples,\n        micro_batch_size=MICRO_BATCH_SIZE,\n    ):\n        rows = []\n\n        for start in range(\n            0,\n            len(samples),\n            int(micro_batch_size),\n        ):\n            chunk = samples[\n                start:start + int(micro_batch_size)\n            ]\n\n            images = torch.stack([\n                self._open_image(\n                    str(sample["image_path"])\n                )\n                for sample in chunk\n            ]).to(\n                self.device,\n                non_blocking=True,\n            )\n\n            probs = (\n                self._predict_tensor(images)\n                .cpu()\n            )\n\n            for i, sample in enumerate(chunk):\n                indices = self._requested_indices(\n                    sample["attribute_names"]\n                )\n\n                row = probs[\n                    i,\n                    indices,\n                ].tolist()\n\n                if len(row) != 40:\n                    raise RuntimeError(\n                        "Prediction row does not contain 40 values."\n                    )\n\n                if not all(\n                    math.isfinite(float(x))\n                    for x in row\n                ):\n                    raise RuntimeError(\n                        "Prediction row contains NaN/Inf."\n                    )\n\n                if not all(\n                    0.0 <= float(x) <= 1.0\n                    for x in row\n                ):\n                    raise RuntimeError(\n                        "Prediction outside [0, 1]."\n                    )\n\n                rows.append([\n                    float(x)\n                    for x in row\n                ])\n\n            del images\n            del probs\n\n        return rows\n\n\ndef _ensure_model():\n    global _MODEL\n\n    if _MODEL is None:\n        _MODEL = PrototypeRuntime(\n            ROOT / "assets" / "model.pt"\n        )\n\n    return _MODEL\n\n\ndef load_model():\n    _ensure_model()\n    return None\n\n\ndef _validate_sample(sample):\n    if not isinstance(sample, dict):\n        raise TypeError(\n            "Each sample must be a dictionary."\n        )\n\n    required = (\n        "image_path",\n        "attribute_names",\n    )\n\n    missing = [\n        key\n        for key in required\n        if key not in sample\n    ]\n\n    if missing:\n        raise KeyError(\n            f"Submission sample is missing keys: {missing}"\n        )\n\n    if len(sample["attribute_names"]) != 40:\n        raise ValueError(\n            "Expected exactly 40 attribute_names."\n        )\n\n\ndef predict_image(sample):\n    _validate_sample(sample)\n\n    model = _ensure_model()\n\n    return model.predict_samples(\n        [sample],\n        micro_batch_size=1,\n    )[0]\n\n\ndef predict_batch(samples):\n    if not isinstance(\n        samples,\n        (list, tuple),\n    ):\n        raise TypeError(\n            "predict_batch expects a list/tuple of samples."\n        )\n\n    if len(samples) == 0:\n        return []\n\n    for sample in samples:\n        _validate_sample(sample)\n\n    model = _ensure_model()\n\n    return model.predict_samples(\n        list(samples),\n        micro_batch_size=MICRO_BATCH_SIZE,\n    )\n'


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
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--skip-runtime-smoke-test", action="store_true")
    return p.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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


def runtime_smoke_test(zip_path, attribute_names):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        unpacked = td / "submission"
        unpacked.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(unpacked)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(unpacked / "run.py"),
            ],
            check=True,
        )

        from PIL import Image

        dummy = td / "smoke.jpg"
        Image.new(
            "RGB",
            (144, 288),
            (127, 127, 127),
        ).save(dummy)

        smoke_script = textwrap.dedent(
            f"""
            import math
            import sys
            from pathlib import Path

            root = Path({str(unpacked)!r})
            sys.path.insert(0, str(root))

            import run

            attrs = {list(attribute_names)!r}
            sample = {{
                "image_path": {str(dummy)!r},
                "attribute_names": attrs,
            }}

            run.load_model()

            one = run.predict_image(sample)
            two = run.predict_batch([sample, sample])

            assert len(one) == 40
            assert len(two) == 2
            assert all(len(row) == 40 for row in two)
            assert all(math.isfinite(float(x)) for x in one)
            assert all(0.0 <= float(x) <= 1.0 for x in one)

            print("RUNTIME_SMOKE_TEST_OK")
            print("BATCH_SIZE=", run.BATCH_SIZE)
            print("MIN=", min(one), "MAX=", max(one))
            """
        )

        env = os.environ.copy()
        env["UPAR_MICRO_BATCH_SIZE"] = "1"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                smoke_script,
            ],
            cwd=unpacked,
            env=env,
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Exported submission runtime smoke test FAILED.\n"
                "STDOUT:\n" + result.stdout
                + "\nSTDERR:\n" + result.stderr
            )

        print(result.stdout.strip())


def main():
    args = parse_args()

    if int(args.micro_batch_size) < 1:
        raise ValueError("--micro-batch-size must be >= 1")

    args.result_dir.mkdir(parents=True, exist_ok=True)

    calibration = safe_torch_load(
        args.calibration,
        map_location="cpu",
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
            raise FileNotFoundError(
                f"Required checkpoint not found: {p}"
            )

        ck = safe_torch_load(
            p,
            map_location="cpu",
        )

        state = (
            ck.get("inference_model_state")
            or ck.get("ema_model_state")
            or ck.get("model_state")
        )

        if state is None:
            raise RuntimeError(
                f"{p} has no inference/model state"
            )

        config = dict(
            ck.get(
                "prototype_config",
                meta.get("prototype_config", {}),
            )
        )

        packaged_members.append({
            "seed": seed,
            "weight": float(weight),
            "prototype_config": config,
            "state_dict": half_state_dict(state),
        })

    if not packaged_members:
        raise RuntimeError("No prototype ensemble members were packaged.")

    if sum(max(0.0, float(x["weight"])) for x in packaged_members) <= 0:
        raise RuntimeError("All calibrated ensemble weights are zero.")

    out = (
        args.result_dir
        / "_submission_prototype_autoensemble_build"
    )

    if out.exists():
        shutil.rmtree(out)

    (out / "assets").mkdir(parents=True, exist_ok=True)

    attribute_names = list(
        calibration.get(
            "attribute_names",
            ATTRIBUTE_NAMES,
        )
    )

    if len(attribute_names) != 40:
        raise RuntimeError(
            "Calibration does not contain exactly 40 attributes."
        )

    package = {
        "format": "upar-hybrid-prototype-autoensemble",
        "members": packaged_members,
        "threshold_logits": threshold_logits,
        "attribute_names": attribute_names,
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
        "attribute_names": attribute_names,
        "members": [
            {
                "seed": x["seed"],
                "weight": x["weight"],
                "prototype_config": x["prototype_config"],
            }
            for x in packaged_members
        ],
        "tta": package["tta"],
        "challenge_batch_size": 64,
        "micro_batch_size": int(args.micro_batch_size),
        "ensemble_space": "raw hybrid logits",
        "calibration": "single post-ensemble threshold vector",
        "extra_logit_bias": float(args.extra_logit_bias),
    }

    (out / "assets" / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    run_py = RUNTIME_SOURCE.replace(
        "__MICRO_BATCH_SIZE__",
        str(int(args.micro_batch_size)),
    )

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
        "UPAR Track-1 PAR submission using ConvNeXt hybrid "
        "positive/negative prototypes, validation-driven ensemble "
        "weighting, post-ensemble calibration, and flip-TTA.\n",
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

    expected = {
        "LICENSE-model.txt",
        "NOTICE.md",
        "assets/config.json",
        "assets/model.pt",
        "metadata.yaml",
        "run.py",
    }

    with zipfile.ZipFile(final) as z:
        names = set(z.namelist())

    if names != expected:
        raise RuntimeError(
            "Unexpected submission archive structure. "
            f"Expected={sorted(expected)} got={sorted(names)}"
        )

    print("Built:", final)
    print("MiB:", final.stat().st_size / 1024 / 1024)
    print("SHA256:", sha256(final))
    print(
        "Members:",
        [
            (x["seed"], round(float(x["weight"]), 4))
            for x in packaged_members
        ],
    )
    print("TTA:", package["tta"])
    print("Challenge BATCH_SIZE:", 64)
    print("Internal micro-batch:", int(args.micro_batch_size))
    print(
        "Calibration Challenge_Avg:",
        package["calibration_metrics"].get("Challenge_Avg"),
    )

    if not args.skip_runtime_smoke_test:
        print(
            "\nRunning exact exported-ZIP runtime smoke test...",
            flush=True,
        )
        runtime_smoke_test(
            final,
            attribute_names,
        )
        print(
            "SUBMISSION_RUNTIME_CHECK: PASS",
            flush=True,
        )
    else:
        print(
            "WARNING: runtime smoke test was skipped.",
            flush=True,
        )


if __name__ == "__main__":
    main()
