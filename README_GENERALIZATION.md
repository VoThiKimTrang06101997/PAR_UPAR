# UPAR Prototype Domain-Robust Retraining

**Goal:** train new Prototype checkpoints, compare against the original strong
Prototype checkpoints, and export only when the retrained candidate passes a
source-domain FIT/AUDIT safeguard. A hidden Codabench score of 0.76–0.78 is a
research target, **not a guaranteed result**.

## Main files (no numbered version suffixes)

- `prototype_par.py`: unchanged architecture, correct 40 attributes, visual
  attribute queries, positive/negative prototype banks and hybrid linear head.
- `train_prototype_generalization.py`: new, actual optimizer-based retraining
  from the previous Prototype checkpoints, with a conservative learning rate,
  masked class-balanced BCE, mild attribute-prototype loss, small domain
  alignment loss, mild hard-negative loss, worst-source excess regularization,
  frozen-original-Prototype response distillation, EMA and robust checkpoint
  selection. Resumes from its own `*_last.pt` when interrupted.
- `calibrate_generalization.py`: compares old/new Prototype models and optional
  ConvNeXt source models using a predeclared small ensemble family; fit/audit
  by stable image filename hashes, with source-domain guards and protected old
  checkpoint candidate.
- `export_generalization.py`: exports selected weights/calibration and refuses
  `--require-trained` if the audit guard did not accept any retrained member.
- `submission_generalization.py`: bundled into Codabench `run.py` via exporter.
  Flip TTA, 40 attributes, GPU micro-batch, self-contained runtime.

## What was fixed from the uploaded precision-trainer notebook

1. The optional retraining path was not the default. This notebook now **trains**
   seed 42 and seed 123 by default.
2. Earlier changes drove false-positive penalties and prototype gates too far;
   regularization is smaller and the previous prototype checkpoint is the
   training initialization and frozen reference teacher.
3. `calibrate_and_save` previously could mutate the actively training model
   with EMA weights after a validation improvement; this is now routed to
   the separate EMA module, preserving optimizer/model consistency.
4. Prototype inference no longer relies on the faulty assumption that domain
   names are dictionaries; this pipeline uses explicit held-domain IDs and
   avoids fragile domain-name lookups during export.
5. Model calibration never sees Codabench hidden labels. FIT/AUDIT are two
   partitions of the **source validation** set, not true unseen-domain proof.

## Intended output

Checkpoints:

- `prototype_generalization_convnext_tiny_seed42_best.pt`
- `prototype_generalization_convnext_tiny_seed123_best.pt`

Source-domain validation:

- `generalization_audit_report.json`
- `generalization_calibration.pt`

Submission if accepted:

- `Submission_PROTOTYPE_GENERALIZATION_TTA.zip`

The ZIP exporter internally unpacks and smoke-tests the *actual* archive via
`load_model()`, `predict_image()` and `predict_batch()`. Look for
`SUBMISSION_RUNTIME_CHECK: PASS` before uploading. This is a runtime test and
not a substitute for the private challenge score.

## Safety notes

- Do not overwrite original `prototype_convnext_tiny_seed*_best.pt` files.
- This source patch is grounded in the user-uploaded GitHub archive and Colab;
  the live GitHub commit may have changed since those snapshots.
- Real GPU training with the full datasets and Codabench private scoring were
  not available inside the file-generation environment.
