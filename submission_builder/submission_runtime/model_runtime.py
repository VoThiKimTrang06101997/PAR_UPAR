from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

# Hard offline mode. The challenge explicitly provides no network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel


def l2n(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class OfflineVAPORCore(nn.Module):
    """Vision-only inference graph.

    Text encoding is NOT performed on the challenge server. The language
    prototypes learned/encoded during training are already stored in the bundle.
    """

    def __init__(self, bundle):
        super().__init__()

        vision_cfg = Siglip2VisionConfig(**bundle["vision_config"])
        self.vision_model = Siglip2VisionModel(vision_cfg)

        missing, unexpected = self.vision_model.load_state_dict(
            bundle["vision_state"],
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Bundled SigLIP2 vision weights do not match runtime architecture. "
                f"missing={missing[:12]}, unexpected={unexpected[:12]}"
            )

        hs = bundle["head_state"]
        hidden_dim = int(bundle["meta"]["hidden_dim"])
        text_dim = int(bundle["meta"]["text_dim"])
        num_attributes = int(bundle["meta"]["num_attributes"])
        num_heads = int(bundle["meta"]["num_heads"])

        self.learned_queries = nn.Parameter(torch.empty(num_attributes, hidden_dim))
        self.semantic_query_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.lang_to_hidden = nn.Linear(text_dim, hidden_dim, bias=False)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(hidden_dim)

        ffn_hidden = int(hs["ffn.1.weight"].shape[0])
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(ffn_hidden, hidden_dim),
        )

        self.visual_pos = nn.Parameter(torch.empty(num_attributes, hidden_dim))
        self.visual_neg = nn.Parameter(torch.empty(num_attributes, hidden_dim))
        self.gate_logits = nn.Parameter(torch.empty(num_attributes, 1))
        self.log_tau = nn.Parameter(torch.empty(()))

        # Register final-shape buffers before loading the head state.
        self.register_buffer(
            "lang_pos",
            torch.empty_like(hs["lang_pos"]),
            persistent=True,
        )
        self.register_buffer(
            "lang_neg",
            torch.empty_like(hs["lang_neg"]),
            persistent=True,
        )
        self.register_buffer(
            "lang_pos_variants",
            torch.empty_like(hs["lang_pos_variants"]),
            persistent=True,
        )
        self.register_buffer(
            "lang_neg_variants",
            torch.empty_like(hs["lang_neg_variants"]),
            persistent=True,
        )

        missing, unexpected = self.load_state_dict(hs, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Bundled VAPOR-PAR head does not match runtime architecture. "
                f"missing={missing[:12]}, unexpected={unexpected[:12]}"
            )

    def language_hidden(self):
        lp = l2n(self.lang_to_hidden(self.lang_pos), -1)
        ln = l2n(self.lang_to_hidden(self.lang_neg), -1)
        return lp, ln

    def forward(self, inputs):
        vision_kwargs = {}
        for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
            if key in inputs:
                vision_kwargs[key] = inputs[key]

        vision_out = self.vision_model(**vision_kwargs)
        patches = vision_out.last_hidden_state

        semantic_dir = l2n(self.lang_pos - self.lang_neg, -1)
        queries = self.learned_queries + self.semantic_query_proj(semantic_dir)
        queries = queries.unsqueeze(0).expand(patches.size(0), -1, -1)

        key_padding_mask = None
        pam = inputs.get("pixel_attention_mask")
        if (
            pam is not None
            and pam.ndim == 2
            and pam.shape[1] == patches.shape[1]
        ):
            key_padding_mask = ~pam.bool()

        attended, _ = self.cross_attn(
            self.query_norm(queries),
            patches,
            patches,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        h = l2n(queries + attended + self.ffn(queries + attended), -1)

        lang_pos_h, lang_neg_h = self.language_hidden()
        gate = torch.sigmoid(self.gate_logits)

        r_pos = l2n(
            gate * self.visual_pos + (1.0 - gate) * lang_pos_h,
            -1,
        )
        r_neg = l2n(
            gate * self.visual_neg + (1.0 - gate) * lang_neg_h,
            -1,
        )

        s_pos = (h * r_pos.unsqueeze(0)).sum(-1)
        s_neg = (h * r_neg.unsqueeze(0)).sum(-1)

        tau = self.log_tau.exp().clamp(0.01, 1.0)
        return (s_pos - s_neg) / tau


class OfflineVAPORPAR:
    def __init__(self, bundle_path):
        self.bundle_path = Path(bundle_path)

        try:
            self.bundle = torch.load(
                self.bundle_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # torch 2.4.1 does not require the weights_only kwarg.
            self.bundle = torch.load(
                self.bundle_path,
                map_location="cpu",
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = OfflineVAPORCore(self.bundle).to(self.device).eval()

        self.processor = Siglip2ImageProcessor.from_dict(
            self.bundle["processor_config"]
        )

        self.internal_attribute_names = list(
            self.bundle["attribute_names"]
        )
        if len(self.internal_attribute_names) != 40:
            raise RuntimeError("Bundle must contain exactly 40 attributes.")

        self.name_to_index = {
            name: i for i, name in enumerate(self.internal_attribute_names)
        }

        # Vector-scaling calibration learned after training.
        calibration = self.bundle.get("calibration")
        if calibration is None:
            self.cal_log_scale = None
            self.cal_bias = None
        else:
            self.cal_log_scale = calibration["log_scale"].float().to(self.device)
            self.cal_bias = calibration["bias"].float().to(self.device)

        # Challenge evaluator thresholds returned probabilities at 0.5.
        # Convert the learned per-attribute threshold t into a logit offset:
        # sigmoid(z - logit(t)) > 0.5  <=>  sigmoid(z) > t.
        thresholds = self.bundle.get("thresholds")
        if thresholds is None:
            self.threshold_logit = None
        else:
            t = torch.as_tensor(
                thresholds,
                dtype=torch.float32,
                device=self.device,
            ).clamp(1e-4, 1.0 - 1e-4)
            self.threshold_logit = torch.logit(t)

    def _apply_calibration_and_threshold_shift(self, logits):
        if self.cal_log_scale is not None:
            logits = (
                logits
                * self.cal_log_scale.exp().clamp(0.25, 4.0)
                + self.cal_bias
            )

        if self.threshold_logit is not None:
            logits = logits - self.threshold_logit

        return logits

    def _open_image(self, path):
        with Image.open(path) as im:
            return im.convert("RGB")

    def _requested_indices(self, attribute_names):
        requested = list(attribute_names)
        if len(requested) != 40:
            raise ValueError(
                f"Expected 40 attributes, got {len(requested)}"
            )

        unknown = [x for x in requested if x not in self.name_to_index]
        if unknown:
            raise KeyError(
                "Challenge requested unknown attributes: "
                + ", ".join(unknown)
            )

        return [self.name_to_index[x] for x in requested]

    @torch.inference_mode()
    def _predict_paths_internal(self, paths, micro_batch_size=16):
        chunks = []

        for start in range(0, len(paths), micro_batch_size):
            part = paths[start:start + micro_batch_size]
            images = [self._open_image(p) for p in part]

            processed = self.processor(
                images=images,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.device, non_blocking=True)
                for k, v in processed.items()
                if torch.is_tensor(v)
            }

            if self.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = self.model(inputs)
                    logits = self._apply_calibration_and_threshold_shift(logits)
                    probs = torch.sigmoid(logits)
            else:
                logits = self.model(inputs)
                logits = self._apply_calibration_and_threshold_shift(logits)
                probs = torch.sigmoid(logits)

            # Make challenge validation robust to tiny numerical excursions.
            probs = torch.nan_to_num(
                probs.float(),
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

            chunks.append(probs.cpu())

        if not chunks:
            return torch.empty(
                (0, len(self.internal_attribute_names)),
                dtype=torch.float32,
            )

        return torch.cat(chunks, dim=0)

    def predict_samples(self, samples, micro_batch_size=16):
        paths = [str(sample["image_path"]) for sample in samples]

        internal_probs = self._predict_paths_internal(
            paths,
            micro_batch_size=micro_batch_size,
        )

        rows = []
        for i, sample in enumerate(samples):
            indices = self._requested_indices(sample["attribute_names"])
            row = internal_probs[i, indices].tolist()

            if len(row) != 40:
                raise RuntimeError("Prediction row does not have 40 probabilities.")
            if not all(np.isfinite(row)):
                raise RuntimeError("Prediction row contains NaN/Inf.")
            if not all(0.0 <= float(x) <= 1.0 for x in row):
                raise RuntimeError("Prediction row contains probability outside [0,1].")

            rows.append([float(x) for x in row])

        return rows
