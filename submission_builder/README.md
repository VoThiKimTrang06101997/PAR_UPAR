# UPAR 2027 Track-1 Code Submission

The final Codabench archive is created from:

`best_calibrated.pt`

and contains:

- `run.py` at archive root;
- `metadata.yaml` copied byte-for-byte from the official example submission;
- `model_runtime.py`;
- `weights/inference_bundle.pt`;
- `vendor/` with the Python packages needed because the evaluator has no network.

## Required API implemented

```python
BATCH_SIZE = 64

def load_model() -> None:
    ...

def predict_image(sample: dict) -> list[float]:
    ...

def predict_batch(samples: list[dict]) -> list[list[float]]:
    ...
```

Every output row has exactly 40 finite probabilities in `[0, 1]`, reordered to
the exact order in `sample["attribute_names"]`.

## Fixed 0.5 challenge threshold

Training stores per-attribute thresholds. The challenge evaluator uses `0.5`.

At inference the bundle shifts each calibrated logit by `logit(t_c)`:

`returned_probability = sigmoid(calibrated_logit - logit(t_c))`

Therefore:

`returned_probability > 0.5`

is equivalent to:

`original_calibrated_probability > t_c`.

This lets the submitted probability API preserve the threshold calibration learned
during development while still obeying the challenge's fixed 0.5 threshold.

## Offline environment

The builder exports the full final SigLIP2 vision encoder. It does not rely on the
compact training checkpoint alone.

The builder vendors a Transformers release targeted to Python 3.11 and extracts
all non-base-image dependencies into `vendor/`.

The challenge-provided torch / torchvision / numpy / Pillow are not replaced.
