## IMPORTANT — true mid-epoch resume fix

The old training loop could load `resume_step.pt` correctly but still iterate the
DataLoader from batch 0 and `continue` until the saved batch. This did **not**
repeat optimizer updates, but it did re-decode/re-preprocess earlier images and
made tqdm look like training restarted.

The fixed code uses `ResumableEpochBatchSampler`. If a checkpoint says:

```text
epoch=0
next_batch_idx=4000
total_batches=8140
```

the DataLoader starts directly from batch 4000 and tqdm starts at about 49%.

`KeyboardInterrupt` now writes an emergency `resume_step.pt`. Periodic checkpoints
default to every 100 optimizer steps. A hard VM reset can only resume from the
last checkpoint that physically reached Google Drive; unsaved work cannot be
reconstructed after the process is already gone.

The SigLIP2 base model still has to be instantiated in a fresh Colab process.
That `Loading weights...` message is model reconstruction, **not training from
epoch 0**. The fixed config caches Hugging Face files under:

`/content/drive/MyDrive/PedestrianAttributeRecognition/HFCache`

so future sessions reuse the downloaded model files.

## UPAR 2027 code submission

The generated archive follows the documented API:

```python
def load_model() -> None: ...
def predict_image(sample: dict) -> list[float]: ...
BATCH_SIZE = 64
def predict_batch(samples: list[dict]) -> list[list[float]]: ...
```

At archive root:

```text
run.py
metadata.yaml
model_runtime.py
weights/inference_bundle.pt
vendor/...
```

`metadata.yaml` is copied byte-for-byte from the official example submission.
The runtime never downloads from the network.

# PAR_UPAR — Final Resume + Offline Codabench Submission

## Resume fix

A mid-epoch checkpoint such as:

`epoch=0, next_batch_idx=4000, global_step=4000`

now resumes by changing the **batch sampler itself**. The DataLoader does not open,
augment, preprocess, or discard batches `0..3999`.

The progress bar starts at approximately:

`4000 / 8140 ≈ 49%`

instead of visually counting from zero.

If both `resume_step.pt` and `last.pt` exist, training inspects both and selects the
checkpoint with the greatest `(global_step, epoch, next_batch_idx)`.

## Submission

After `best_calibrated.pt` exists:

```bash
python submission_builder/build_submission.py \
  --repo-dir /content/PAR_UPAR \
  --checkpoint "/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints/best_calibrated.pt" \
  --metadata-yaml "/path/to/example_submission/metadata.yaml" \
  --output "/content/drive/MyDrive/Pedestrian Attribute Recognition/Results/PAR_UPAR_submission.zip"
```

The final ZIP contains `run.py` and `metadata.yaml` at the archive root and implements
the exact UPAR 2027 Track-1 `load_model`, `predict_image`, and `predict_batch` API.

---

# PAR_UPAR — Resume-Fixed VAPOR-PAR + Offline Codabench Submission

This revision fixes the mid-epoch resume behavior visible in Colab and adds the offline code-submission builder.

## What was wrong with resume?

The old code correctly loaded `resume_step.pt`, e.g. `epoch=0, batch=4000, global_step=4000`, but then recreated the full DataLoader from batch 0 and executed:

```python
if batch_idx < resume_batch:
    continue
```

That does not redo optimizer updates, but DataLoader workers still decode/preprocess batches 0..3999. The progress bar therefore starts at 0 and makes it look like training restarted.

## Resume fix

Training now uses `ResumableEpochBatchSampler`.

For a checkpoint at batch 4000 / 8140:

- sampler reconstructs the same deterministic epoch permutation;
- it slices directly from sample `4000 * batch_size`;
- batches 0..3999 are never loaded;
- tqdm begins at `4000/8140` rather than 0;
- optimizer, scheduler, AMP scaler, global step, model weights, thresholds and calibration state are restored;
- `KeyboardInterrupt` triggers an emergency checkpoint save at the last safe optimizer boundary;
- a Colab/runtime crash still falls back to the latest periodic `resume_step.pt`.

Existing older `resume_step.pt` files are supported.

## Drive paths

Checkpoints:

```text
/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints/
```

Results:

```text
/content/drive/MyDrive/Pedestrian Attribute Recognition/Results/
```

## Inspect current checkpoint

```bash
python inspect_checkpoint.py
```

## Train / resume

Resume is the default:

```bash
python train_colab.py \
  --epochs 25 \
  --batch-size 12 \
  --eval-batch-size 24 \
  --num-workers 2 \
  --save-every-steps 500 \
  --log-every-steps 50
```

Use `--restart` only when you intentionally want a new run.

## Offline Codabench code submission

The challenge runner has no network access and requires `run.py` at ZIP root. After training completes and `best_calibrated.pt` exists:

```bash
python submission_builder/build_submission.py \
  --repo-dir /content/PAR_UPAR \
  --checkpoint "/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints/best_calibrated.pt" \
  --official-starter-dir /content/UPAR-Challenge-2027 \
  --output "/content/drive/MyDrive/Pedestrian Attribute Recognition/Results/PAR_UPAR_submission.zip"
```

The builder reconstructs the compact training checkpoint, exports the full SigLIP2 vision encoder plus VAPOR-PAR head/calibration into an offline bundle, vendors the required Python runtime packages, and creates a ZIP whose archive root contains `run.py`.

The exact callable contract of the organizer's `run.py` should still be checked against the latest example submission/Submission page. The supplied wrapper exposes common `load_model`, `predict`, `predict_batch`, `inference`, `run`, and `Model` aliases.
