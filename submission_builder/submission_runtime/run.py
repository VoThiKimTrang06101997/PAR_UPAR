"""UPAR Challenge 2027 — Track 1 submission entry point.

Required challenge API:
    load_model() -> None
    predict_image(sample: dict) -> list[float]
    predict_batch(samples: list[dict]) -> list[list[float]]

The evaluator provides absolute paths in sample["image_path"].
No network access is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

from model_runtime import OfflineVAPORPAR

# The ingestion program may batch up to this many samples per API call.
# The model runtime itself uses a smaller GPU micro-batch to avoid OOM.
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 16

_MODEL = None


def _ensure_model() -> OfflineVAPORPAR:
    global _MODEL
    if _MODEL is None:
        _MODEL = OfflineVAPORPAR(ROOT / "weights" / "inference_bundle.pt")
    return _MODEL


def load_model() -> None:
    """Called once before inference. Load all offline weights here."""
    _ensure_model()
    return None


def _validate_sample(sample: dict) -> None:
    required = ("image_path", "attribute_names")
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError(f"Submission sample is missing keys: {missing}")
    if len(sample["attribute_names"]) != 40:
        raise ValueError(
            f"Expected 40 attribute_names, got {len(sample['attribute_names'])}"
        )


def predict_image(sample: dict) -> list[float]:
    """Return exactly 40 probabilities in sample['attribute_names'] order."""
    _validate_sample(sample)
    model = _ensure_model()
    rows = model.predict_samples([sample], micro_batch_size=1)
    return rows[0]


def predict_batch(samples: list[dict]) -> list[list[float]]:
    """Return one row of 40 probabilities per sample, preserving sample order."""
    if not isinstance(samples, (list, tuple)):
        raise TypeError("predict_batch expects a list of sample dictionaries.")
    if len(samples) == 0:
        return []
    for sample in samples:
        _validate_sample(sample)

    model = _ensure_model()
    return model.predict_samples(
        list(samples),
        micro_batch_size=MICRO_BATCH_SIZE,
    )
