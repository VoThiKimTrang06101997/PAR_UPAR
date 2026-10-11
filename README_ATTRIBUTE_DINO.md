# UPAR Track 1 — Attribute-balanced Hybrid Prototype + DINOv2-S/14

**Purpose:** evaluate an additional, genuinely pretrained DINOv2-S feature branch and an improved ConvNeXt Hybrid Prototype branch against the existing protected Prototype anchor. No private-test labels are accessed. Achieving 0.78 is a research target, not a verified result.

## Added files
- `dinov2_par.py`: standalone PyTorch DINOv2-S/14 implementation with pretrained HF tensor conversion in training only. 40 attribute-specific spatial queries, positive/negative prototypes, globally gated classifier; no transformers dependency in Codabench ZIP.
- `train_dinov2_par.py`: supports `--encoder dinov2` and `--encoder convnext`. Masked Balanced ASL, domain mixing, class-conditional supervised contrastive, cross-domain prototype alignment, prediction consistency, spatial queries, separate resume/best checkpoints. ConvNeXt branch fine-tunes an original Prototype into a new *separate* checkpoint. Only train domains used in LODO (`--heldout-domain`). Val FIT only used to choose epochs.
- `prepare_upar_source.py`: check all annotated source validation images, use original official UPAR data downloader if needed. Does not delete populated image directories.

## Updated source files
- `calibrate_shift_robust.py`: include `--dinov2-checkpoint-dir` and `--robust-checkpoint-dir`. Source FIT/AUDIT ensemble candidate discovery, locked audit check, per-attribute thresholds, cache and TTA, optional original Strong checkpoints. If a challenger is not accepted, return protected anchor. Save separate experimental calibration; no claim it passed the audit.
- `submission_generalization.py`: embeds pure-PyTorch DINOv2 model so exported ZIP does not need transformers, HF or internet. DINO trained at 252×126; 288×144 original input is bilinear-interpolated to DINO size during calibration *and* runtime. Original Proto retains its original input.
- `export_generalization.py`: includes model type `dinov2` and `robust` checkpoint architecture/config in the ZIP. Executes load_model / predict_image / predict_batch smoke tests from exact ZIP and validates 40 predictions.
- Compatibility files: `prototype_par.py`, `calibrate_generalization.py`, `domain_contrastive.py`, `upar_progress.py`, `inspect_submission.py`.

## Colab workflow
1. Clone https://github.com/VoThiKimTrang06101997/PAR_UPAR and apply the embedded patch (not reusing a stale clone).
2. Read official UPAR source annotations and *verify the source images exist*.
3. Train one robust ConvNeXt Hybrid student with new loss (seed42), and one independent DINOv2-S student (seed42). Optionally add seed123 **only if the first round improves LODO/audit**. No need to rerun 18 old LODO trials.
4. Calibrate on source VAL FIT, audit once on held-out source VAL AUDIT, compare with protected prototype seed42/123.
5. Export `Submission_UPAR_AttributeDINO_AuditSelected.zip` (audited choice) and if distinct `Submission_UPAR_AttributeDINO_Experimental.zip` (untested exploration). Do **not** assume that the experimental bundle is better.

## Safe interpretation
- Private Codabench benchmark ~0.7438 remains the last verified target-domain score.
- Source VAL/FIT and source AUDIT can overestimate performance on unseen domains.
- `run.py` is standalone for the exact Python/PyTorch/torchvision runtime. The source-model converter requires `transformers` only when downloading initial pretrained weights in Colab.
- Model `facebook/dinov2-small` model card specifies Apache-2.0. Confirm challenge rules allow use of this pretrained source and comply with license/attribution requirements.
- Epoch selection uses only source FIT; for stronger evidence train 3 separate held-out-domain folds, compare baseline and new branches on each held-out domain, and do not use held-out metrics for model selection.
