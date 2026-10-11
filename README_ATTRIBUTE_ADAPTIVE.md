# UPAR Attribute-Adaptive Ensemble

This extension is designed for Track 1 PAR, 40 attributes, using the existing Hybrid Prototype / SupCon / DINOv2 checkpoints. It makes **no claim** of >0.78 private Codabench performance without a successful official scoring run.

## Goal

The previously scored `0.750279` private submission is retained as a protected reference. The new calibration learns a regularized **K-by-40 ensemble weight matrix** rather than a single K-vector shared by all attributes. Images, model checkpoints, and original calibration remain unchanged. The optimizer uses masked, domain/class-balanced binary cross-entropy on public source FIT. Penalties and shrinkage are selected on an **internal FIT holdout**. A single resulting candidate is then compared with the saved reference on source AUDIT.

The result is exported with a standalone Codabench `run.py`, containing no imports from the participant repository. Validation inference may be reused using the existing `shift_predictions_*.npz` caches; the cache signature checks checkpoint and inference implementation.

## New files

- `calibrate_attribute_adaptive.py`: FIT-only attribute gate optimization, inner-FIT selection, locked source AUDIT and safe/experimental calibration outputs.
- `ensure_upar_prediction_cache.py`: checks/produces existing per-model source validation prediction caches atomically, with live per-batch tqdm.
- `submission_attribute_adaptive.py`: self-contained Codabench runtime supporting Kx40 gates as well as legacy scalar weights.
- `export_attribute_adaptive.py`: exact six-member ZIP package + isolated runtime smoke test.
- `README_ATTRIBUTE_ADAPTIVE.md`: these instructions.

## Updated supporting code

The included `dinov2_par.py` accepts current and legacy Hugging Face DINOv2 pretrained attention weight keys; other existing source files from the user's provided notebook are packaged to make a fresh Colab checkout reproducible. Original trained checkpoints in Google Drive are **never overwritten** by this calibration/export pipeline.

## Run order

1. Open `PAR_UPAR_AttributeAdaptive.ipynb`, activate Colab GPU.
2. Mount Google Drive and clone the user's `PAR_UPAR` repository.
3. Apply embedded source patch, then verify existing `AttributeDINO/attribute_dino_audit_calibration.pt` and source-validation images. No training required.
4. Reuse or refresh only necessary cached model logits.
5. Run attribute-wise calibration. Read `AttributeAdaptive/attribute_adaptive_report.json` to decide whether to submit the audit-selected ZIP or optional experimental ZIP.
6. Export exact Codabench ZIP and folder, test `load_model` / `predict_image` / `predict_batch` on *real source-validation images*. Upload ZIP only.

Expected Drive path:
`MyDrive/PedestrianAttributeRecognition/Results/AttributeAdaptive/` (calibrations, report, log), and `Results/Submission_UPAR_AttributeAdaptive_AuditSelected.zip` or `..._Experimental.zip`.

## Limitations and integrity

- Source FIT/AUDIT results are **not** private Codabench scores; repeatedly developing methods after seeing source AUDIT may overfit this split.
- Local smoke tests do not replicate Codabench container timeouts, memory limits, or dependency versions.
- Training a second independent DINO model may help but is an expensive separate experiment and is not automatically mixed into the protected base calibration.
- Confirm current challenge rules permit all pretrained weights, ensembles, and using source validation labels for posthoc calibration.
- If hidden score is unchanged or ingestion fails, download the *ingestion* log; a successful local smoke test is insufficient to diagnose it.
- The linked GitHub repository was not fetched live in the artifact creation environment. The implementation was verified against the uploaded notebook's embedded source snapshot and an available archive of the user repository, not against unseen new commits.
