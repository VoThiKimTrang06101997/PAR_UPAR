from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

import torch


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
        default="Submission_PROTOTYPE_PRECISION_TTA.zip",
    )
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=16,
    )
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
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def half_state_dict(state):
    out = {}

    for key, value in state.items():
        tensor = value.detach().cpu()

        if torch.is_floating_point(
            tensor
        ):
            tensor = tensor.half()

        out[key] = tensor

    return out


def checkpoint_path(root, prefix, seed):
    return (
        root
        / f"{prefix}_seed{seed}_best.pt"
    )


RUNTIME = r"""from __future__ import annotations

import gc
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny


ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 64
MICRO_BATCH_SIZE = max(
    1,
    int(
        os.environ.get(
            "UPAR_MICRO_BATCH_SIZE",
            "__MICRO_BATCH_SIZE__",
        )
    ),
)

_MODEL = None


def _safe_torch_load(path, map_location="cpu"):
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


def _logit(p):
    p = min(
        max(float(p), 1e-5),
        1.0 - 1e-5,
    )
    return math.log(
        p / (1.0 - p)
    )


class HybridPrototypePAR(nn.Module):
    def __init__(
        self,
        num_attributes=40,
        prototype_dim=256,
        prototype_temperature=0.20,
        prototype_gate_init=0.35,
        prototype_gate_min=0.08,
        prototype_gate_max=0.92,
        prototype_ema_momentum=0.95,
        prototype_ema_mix=0.25,
    ):
        super().__init__()

        self.num_attributes = int(
            num_attributes
        )
        self.prototype_dim = int(
            prototype_dim
        )
        self.prototype_temperature = float(
            prototype_temperature
        )
        self.prototype_gate_min = float(
            prototype_gate_min
        )
        self.prototype_gate_max = float(
            prototype_gate_max
        )
        self.prototype_ema_momentum = float(
            prototype_ema_momentum
        )
        self.prototype_ema_mix = float(
            prototype_ema_mix
        )

        base = convnext_tiny(
            weights=None
        )

        self.backbone = base
        self.backbone_dim = int(
            base.classifier[2].in_features
        )
        self.backbone.classifier[2] = (
            nn.Identity()
        )

        self.global_dropout = nn.Dropout(
            0.22
        )
        self.linear_head = nn.Linear(
            self.backbone_dim,
            self.num_attributes,
        )

        self.token_norm = nn.LayerNorm(
            self.backbone_dim
        )
        self.token_proj = nn.Linear(
            self.backbone_dim,
            self.prototype_dim,
            bias=False,
        )

        self.attribute_queries = (
            nn.Parameter(
                torch.zeros(
                    self.num_attributes,
                    self.prototype_dim,
                )
            )
        )

        self.attribute_norm = nn.LayerNorm(
            self.prototype_dim
        )

        self.prototype_pos = nn.Parameter(
            torch.zeros(
                self.num_attributes,
                self.prototype_dim,
            )
        )

        self.prototype_neg = nn.Parameter(
            torch.zeros(
                self.num_attributes,
                self.prototype_dim,
            )
        )

        self.register_buffer(
            "prototype_pos_ema",
            torch.zeros(
                self.num_attributes,
                self.prototype_dim,
            ),
        )

        self.register_buffer(
            "prototype_neg_ema",
            torch.zeros(
                self.num_attributes,
                self.prototype_dim,
            ),
        )

        self.register_buffer(
            "prototype_pos_updates",
            torch.zeros(
                self.num_attributes,
                dtype=torch.long,
            ),
        )

        self.register_buffer(
            "prototype_neg_updates",
            torch.zeros(
                self.num_attributes,
                dtype=torch.long,
            ),
        )

        self.prototype_gate_logits = (
            nn.Parameter(
                torch.full(
                    (
                        self.num_attributes,
                    ),
                    _logit(
                        prototype_gate_init
                    ),
                    dtype=torch.float32,
                )
            )
        )

    def effective_prototypes(self):
        learned_pos = F.normalize(
            self.prototype_pos,
            dim=-1,
        )
        learned_neg = F.normalize(
            self.prototype_neg,
            dim=-1,
        )
        ema_pos = F.normalize(
            self.prototype_pos_ema,
            dim=-1,
        )
        ema_neg = F.normalize(
            self.prototype_neg_ema,
            dim=-1,
        )

        mix = float(
            self.prototype_ema_mix
        )

        pos = F.normalize(
            (1.0 - mix)
            * learned_pos
            + mix
            * ema_pos,
            dim=-1,
        )

        neg = F.normalize(
            (1.0 - mix)
            * learned_neg
            + mix
            * ema_neg,
            dim=-1,
        )

        return pos, neg

    def components(self, x):
        fmap = self.backbone.features(
            x
        )

        pooled = self.backbone.avgpool(
            fmap
        )
        pooled = self.backbone.classifier[
            0
        ](
            pooled
        )
        global_feature = pooled.flatten(
            1
        )

        linear_logits = self.linear_head(
            self.global_dropout(
                global_feature
            )
        )

        tokens = fmap.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        tokens = self.token_norm(
            tokens
        )
        tokens = self.token_proj(
            tokens
        )
        tokens = tokens.flatten(
            1,
            2,
        )

        q = F.normalize(
            self.attribute_queries,
            dim=-1,
        )
        k = F.normalize(
            tokens,
            dim=-1,
        )

        attn_logits = torch.einsum(
            "bnd,cd->bcn",
            k,
            q,
        ) / 0.10

        attn = torch.softmax(
            attn_logits,
            dim=-1,
        )

        features = torch.einsum(
            "bcn,bnd->bcd",
            attn,
            tokens,
        )

        features = self.attribute_norm(
            features
        )
        features = F.normalize(
            features,
            dim=-1,
        )

        pos, neg = (
            self.effective_prototypes()
        )

        pos_sim = torch.einsum(
            "bcd,cd->bc",
            features,
            pos,
        )
        neg_sim = torch.einsum(
            "bcd,cd->bc",
            features,
            neg,
        )

        proto_logits = (
            pos_sim - neg_sim
        ) / max(
            float(
                self.prototype_temperature
            ),
            1e-4,
        )

        learned_gate = torch.sigmoid(
            self.prototype_gate_logits
        ).reshape(
            1,
            -1,
        )

        return (
            linear_logits,
            proto_logits,
            learned_gate,
        )


def _sanitize_config(config, state):
    config = dict(
        config or {}
    )

    if "linear_head.weight" in state:
        config[
            "num_attributes"
        ] = int(
            state[
                "linear_head.weight"
            ].shape[0]
        )

    if "attribute_queries" in state:
        config[
            "prototype_dim"
        ] = int(
            state[
                "attribute_queries"
            ].shape[1]
        )

    defaults = {
        "num_attributes": 40,
        "prototype_dim": 256,
        "prototype_temperature": 0.20,
        "prototype_gate_init": 0.35,
        "prototype_gate_min": 0.08,
        "prototype_gate_max": 0.92,
        "prototype_ema_momentum": 0.95,
        "prototype_ema_mix": 0.25,
    }

    return {
        key: config.get(
            key,
            default,
        )
        for key, default
        in defaults.items()
    }


class Runtime:
    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        bundle = _safe_torch_load(
            ROOT
            / "assets"
            / "model.pt",
            map_location="cpu",
        )

        self.attributes = list(
            bundle[
                "attribute_names"
            ]
        )

        self.attribute_to_index = {
            name: index
            for index, name
            in enumerate(
                self.attributes
            )
        }

        self.height = int(
            bundle.get(
                "image_height",
                288,
            )
        )
        self.width = int(
            bundle.get(
                "image_width",
                144,
            )
        )

        self.tta = str(
            bundle.get(
                "tta",
                "flip",
            )
        )

        self.threshold_logits = (
            torch.as_tensor(
                bundle[
                    "threshold_logits"
                ],
                dtype=torch.float32,
            )
            .reshape(
                1,
                -1,
            )
            .to(
                self.device
            )
        )

        self.transform = transforms.Compose([
            transforms.Resize(
                (
                    self.height,
                    self.width,
                ),
                interpolation=(
                    transforms.InterpolationMode.BICUBIC
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [
                    0.485,
                    0.456,
                    0.406,
                ],
                [
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ])

        self.models = []
        self.weights = []
        self.scale_vectors = []
        self.gate_maxes = []

        for member in bundle[
            "members"
        ]:
            weight = float(
                member[
                    "weight"
                ]
            )

            if weight <= 0.0:
                member[
                    "state_dict"
                ] = None
                continue

            state = member[
                "state_dict"
            ]

            config = _sanitize_config(
                member.get(
                    "prototype_config",
                    {},
                ),
                state,
            )

            model = HybridPrototypePAR(
                **config
            )

            model.load_state_dict(
                state,
                strict=True,
            )

            member[
                "state_dict"
            ] = None
            del state

            model = model.to(
                self.device
            ).eval()

            scale_vector = torch.as_tensor(
                member[
                    "gate_scale_vector"
                ],
                dtype=torch.float32,
                device=self.device,
            ).reshape(
                1,
                -1,
            )

            self.models.append(
                model
            )
            self.weights.append(
                weight
            )
            self.scale_vectors.append(
                scale_vector
            )
            self.gate_maxes.append(
                float(
                    member[
                        "gate_max"
                    ]
                )
            )

        del bundle
        gc.collect()

        if (
            self.device.type
            == "cuda"
        ):
            torch.cuda.empty_cache()

        if not self.models:
            raise RuntimeError(
                "No active models"
            )

        total = sum(
            self.weights
        )

        self.weights = [
            float(
                weight
                / total
            )
            for weight
            in self.weights
        ]

    def _load_image(self, path):
        with Image.open(
            path
        ) as image:
            return self.transform(
                image.convert(
                    "RGB"
                )
            )

    def _model_logits(
        self,
        model,
        x,
        scale_vector,
        gate_max,
    ):
        def one_view(view):
            (
                linear,
                proto,
                learned_gate,
            ) = model.components(
                view
            )

            gate = torch.clamp(
                learned_gate
                * scale_vector,
                min=0.0,
                max=float(
                    gate_max
                ),
            )

            return (
                (1.0 - gate)
                * linear
                + gate
                * proto
            )

        if (
            self.device.type
            == "cuda"
        ):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = one_view(
                    x
                )

                if (
                    self.tta
                    == "flip"
                ):
                    logits = 0.5 * (
                        logits
                        + one_view(
                            torch.flip(
                                x,
                                dims=[3],
                            )
                        )
                    )
        else:
            logits = one_view(
                x
            )

            if (
                self.tta
                == "flip"
            ):
                logits = 0.5 * (
                    logits
                    + one_view(
                        torch.flip(
                            x,
                            dims=[3],
                        )
                    )
                )

        return logits.float()

    @torch.inference_mode()
    def predict(
        self,
        samples,
    ):
        rows = []

        for start in range(
            0,
            len(samples),
            MICRO_BATCH_SIZE,
        ):
            chunk = samples[
                start:
                start
                + MICRO_BATCH_SIZE
            ]

            x = torch.stack([
                self._load_image(
                    sample[
                        "image_path"
                    ]
                )
                for sample
                in chunk
            ]).to(
                self.device,
                non_blocking=True,
            )

            total = None

            for (
                model,
                weight,
                scale_vector,
                gate_max,
            ) in zip(
                self.models,
                self.weights,
                self.scale_vectors,
                self.gate_maxes,
            ):
                logits = (
                    self._model_logits(
                        model,
                        x,
                        scale_vector,
                        gate_max,
                    )
                    * float(
                        weight
                    )
                )

                total = (
                    logits
                    if total is None
                    else total
                    + logits
                )

            probs = torch.sigmoid(
                total
                - self.threshold_logits
            )

            probs = torch.nan_to_num(
                probs.float(),
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            ).clamp(
                0.0,
                1.0,
            ).cpu()

            for i, sample in enumerate(
                chunk
            ):
                requested = list(
                    sample[
                        "attribute_names"
                    ]
                )

                if len(
                    requested
                ) != 40:
                    raise ValueError(
                        "Expected 40 attributes"
                    )

                indices = [
                    self.attribute_to_index[
                        name
                    ]
                    for name
                    in requested
                ]

                row = probs[
                    i,
                    indices,
                ].tolist()

                if len(
                    row
                ) != 40:
                    raise RuntimeError(
                        "Invalid prediction width"
                    )

                if not all(
                    math.isfinite(
                        float(v)
                    )
                    for v
                    in row
                ):
                    raise RuntimeError(
                        "NaN/Inf output"
                    )

                rows.append([
                    float(v)
                    for v
                    in row
                ])

        return rows


def _ensure_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = Runtime()

    return _MODEL


def load_model():
    _ensure_model()
    return None


def predict_image(sample):
    return _ensure_model().predict(
        [sample]
    )[0]


def predict_batch(samples):
    if not samples:
        return []

    return _ensure_model().predict(
        list(samples)
    )
"""


def runtime_smoke_test(
    zip_path,
    attribute_names,
):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        with zipfile.ZipFile(
            zip_path
        ) as z:
            z.extractall(
                td
            )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(
                    td
                    / "run.py"
                ),
            ],
            check=True,
        )

        from PIL import Image

        image_path = (
            td
            / "smoke.jpg"
        )

        Image.new(
            "RGB",
            (
                144,
                288,
            ),
            (
                127,
                127,
                127,
            ),
        ).save(
            image_path
        )

        script = textwrap.dedent(
            f"""
            import math
            import sys
            sys.path.insert(
                0,
                {str(td)!r},
            )
            import run

            sample = {{
                "image_path": {str(image_path)!r},
                "attribute_names": {list(attribute_names)!r},
            }}

            run.load_model()
            a = run.predict_image(sample)
            b = run.predict_batch([sample, sample])

            assert len(a) == 40
            assert len(b) == 2
            assert all(len(x) == 40 for x in b)
            assert all(math.isfinite(float(x)) for x in a)
            assert all(0.0 <= float(x) <= 1.0 for x in a)

            print("RUNTIME_SMOKE_TEST_OK")
            """
        )

        env = os.environ.copy()
        env[
            "UPAR_MICRO_BATCH_SIZE"
        ] = "1"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
            ],
            cwd=td,
            env=env,
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Runtime smoke test failed.\n"
                + result.stdout
                + "\n"
                + result.stderr
            )

        print(
            result.stdout.strip()
        )


def main():
    args = parse_args()

    calibration = safe_torch_load(
        args.calibration,
        map_location="cpu",
    )

    weights = torch.as_tensor(
        calibration[
            "weights"
        ],
        dtype=torch.float32,
    ).reshape(
        -1
    )

    members = []

    for metadata, weight in zip(
        calibration[
            "members"
        ],
        weights.tolist(),
    ):
        # Do not package an endpoint member that calibration assigned zero
        # weight. This reduces ZIP size, host RAM and GPU initialization cost.
        if float(weight) <= 1e-8:
            continue

        seed = int(
            metadata[
                "seed"
            ]
        )

        prefix = str(
            metadata[
                "checkpoint_prefix"
            ]
        )

        path = checkpoint_path(
            args.checkpoint_dir,
            prefix,
            seed,
        )

        ck = safe_torch_load(
            path,
            map_location="cpu",
        )

        state = (
            ck.get(
                "inference_model_state"
            )
            or ck.get(
                "ema_model_state"
            )
            or ck.get(
                "model_state"
            )
        )

        if state is None:
            raise RuntimeError(
                f"No model state in {path}"
            )

        groups = calibration[
            "attribute_groups"
        ]

        scale_vector = [
            1.0
            for _ in range(
                40
            )
        ]

        for name, indices in groups.items():
            scale = float(
                metadata[
                    "group_scales"
                ][
                    name
                ]
            )

            for index in indices:
                scale_vector[
                    int(index)
                ] = scale

        config = dict(
            metadata.get(
                "prototype_config",
                ck.get(
                    "prototype_config",
                    {},
                ),
            )
        )

        config.setdefault(
            "prototype_gate_min",
            0.08,
        )
        config.setdefault(
            "prototype_gate_max",
            0.92,
        )

        members.append({
            "seed": seed,
            "weight": float(weight),
            "checkpoint_prefix": prefix,
            "prototype_config": config,
            "gate_scale_vector": scale_vector,
            "gate_max": float(
                metadata[
                    "gate_max"
                ]
            ),
            "state_dict": half_state_dict(
                state
            ),
        })

    output_dir = (
        args.result_dir
        / "_prototype_precision_submission"
    )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    (
        output_dir
        / "assets"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = {
        "format": "upar-prototype-precision",
        "members": members,
        "threshold_logits": calibration[
            "threshold_logits"
        ],
        "attribute_names": calibration[
            "attribute_names"
        ],
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
        "tta": calibration.get(
            "tta",
            "flip",
        ),
        "validation_metrics": calibration.get(
            "metrics",
            {},
        ),
    }

    torch.save(
        bundle,
        output_dir
        / "assets"
        / "model.pt",
    )

    (
        output_dir
        / "assets"
        / "config.json"
    ).write_text(
        json.dumps(
            {
                "format": bundle[
                    "format"
                ],
                "members": [
                    {
                        "seed": member[
                            "seed"
                        ],
                        "weight": member[
                            "weight"
                        ],
                        "checkpoint_prefix": member[
                            "checkpoint_prefix"
                        ],
                        "gate_max": member[
                            "gate_max"
                        ],
                    }
                    for member
                    in members
                ],
                "tta": bundle[
                    "tta"
                ],
                "challenge_batch_size": 64,
                "micro_batch_size": int(
                    args.micro_batch_size
                ),
                "validation_metrics": bundle[
                    "validation_metrics"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    run_py = RUNTIME.replace(
        "__MICRO_BATCH_SIZE__",
        str(
            int(
                args.micro_batch_size
            )
        ),
    )

    (
        output_dir
        / "run.py"
    ).write_text(
        run_py,
        encoding="utf-8",
    )

    (
        output_dir
        / "metadata.yaml"
    ).write_text(
        "name: UPAR Prototype Precision TTA\n"
        "task: pedestrian_attribute_recognition\n"
        "framework: pytorch\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "NOTICE.md"
    ).write_text(
        "Hybrid positive/negative prototype PAR with "
        "group-wise prototype-strength calibration, "
        "precision-aware LODO thresholds, weighted ensemble, "
        "and horizontal-flip TTA.\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "LICENSE-model.txt"
    ).write_text(
        "Participant-trained weights. Torchvision components "
        "retain their upstream licenses.\n",
        encoding="utf-8",
    )

    args.result_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        for path in output_dir.rglob(
            "*"
        ):
            if (
                path.is_file()
                and "__pycache__"
                not in path.parts
                and path.suffix
                not in {
                    ".pyc",
                    ".pyo",
                }
            ):
                z.write(
                    path,
                    path.relative_to(
                        output_dir
                    ),
                )

    print(
        "Built:",
        final,
    )
    print(
        "MiB:",
        final.stat().st_size
        / 1024
        / 1024,
    )
    print(
        "SHA256:",
        sha256(
            final
        ),
    )
    print(
        "Validation:",
        json.dumps(
            bundle[
                "validation_metrics"
            ],
            indent=2,
        ),
    )

    runtime_smoke_test(
        final,
        bundle[
            "attribute_names"
        ],
    )

    print(
        "SUBMISSION_RUNTIME_CHECK: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
