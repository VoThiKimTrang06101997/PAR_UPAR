# Hybrid Prototype PAR Patch

This patch adds a prototype-based competition path on top of the current
ConvNeXt AutoEnsemble pipeline without deleting the existing baseline.

## Why this patch

The previous leaderboard pipeline uses:

    Image -> ConvNeXt-Tiny -> global feature -> Linear(40)

The new prototype path keeps that reliable linear classifier as a safety branch,
then adds:

    ConvNeXt spatial features
      -> 40 attribute-specific queries
      -> 40 attribute features
      -> positive / negative prototypes
      -> cosine prototype margin

The final per-attribute logit is a learnable hybrid of the linear and prototype
branches.

This is deliberately safer than replacing the classifier with a pure prototype
head in one step.


## Relationship to the existing `vapor_par/` code

The repository already contains an experimental VAPOR-PAR/SigLIP prototype
implementation under `vapor_par/`.  The uploaded AutoEnsemble Colab does not
train that path; it trains `strong_par.py` instead.  This patch therefore adds
a lighter ConvNeXt prototype path that is directly compatible with the current
competition training, checkpoint, calibration, TTA and submission workflow.

## Prototype components

### Attribute-specific queries

Each of the 40 attributes has a learnable query. It attends over the final
ConvNeXt spatial feature map and produces one feature vector per attribute.

### Positive / negative prototypes

Each attribute owns a learnable positive and negative prototype.

The prototype logit is:

    cosine(h_c, P_c+) - cosine(h_c, P_c-)

scaled by a temperature.

### Prototype EMA

Labeled positive and negative batch features update stable EMA prototypes.
The deployed prototype is a mixture of the learned prototype parameter and
the corresponding EMA prototype buffer.

### Hybrid safety gate

The deployed logit is:

    (1 - gate_c) * linear_logit_c + gate_c * prototype_logit_c

The gate is learned independently for each attribute. This lets the model keep
the strong baseline behavior on attributes where prototypes do not help.

### Warm start from the current strong classifier

If an existing checkpoint such as:

    strong_convnext_tiny_seed42_best.pt

is available, the prototype model copies:

- ConvNeXt feature weights;
- ConvNeXt normalization weights;
- the existing 40-label linear classifier.

The existing linear classifier directions are then orthogonally projected into
prototype space to initialize:

- attribute queries;
- positive prototypes;
- negative prototypes.

Therefore prototype training starts from the current strong solution instead of
discarding it.

### Cross-domain prototype alignment

During training, attribute-specific positive and negative feature centroids are
estimated inside each mini-batch for Market1501, PA100K and PETA.

When the same attribute/class is observed in at least two source domains, their
centroids are softly aligned to a shared direction.

This loss has a small weight and starts only after a warm-up period.

## Training losses

The total training loss contains:

1. masked class-balanced BCE on the hybrid output;
2. a small prototype-margin BCE auxiliary loss;
3. a small cross-domain prototype alignment loss;
4. a small positive/negative prototype separation loss;
5. optional low-weight response distillation from the existing ConvNeXt-Small
   teacher.

The prototype and alignment terms are ramped in after warm-up.

## Existing robustness mechanisms retained

The patch intentionally keeps the stronger parts of the previous pipeline:

- full-body-preserving augmentation;
- tempered source-domain sampling;
- class-balanced BCE and unknown-label masking;
- AdamW;
- mixed precision;
- gradient clipping;
- model EMA;
- top-k checkpoint weight soup with quality gate;
- source-domain robust model selection;
- LODO threshold calibration;
- horizontal-flip TTA;
- validation-driven ensemble weighting;
- post-ensemble calibration.

## Files added

- `prototype_par.py`
- `train_prototype_par.py`
- `calibrate_prototype_ensemble.py`
- `export_prototype_ensemble.py`
- `README_PROTOTYPE.md`

No existing baseline file is deleted, so the old AutoEnsemble path remains
available for direct comparison.

## Checkpoints

Training writes:

    prototype_convnext_tiny_seed42_best.pt
    prototype_convnext_tiny_seed123_best.pt

The ensemble calibration is stored as:

    prototype_ensemble_calibration.pt

Final submission:

    Submission_PROTOTYPE_AUTOENSEMBLE_TTA.zip

## Recommended validation decision

Do not assume prototype learning must beat the old model.

Before submission, compare:

- calibrated Challenge_Avg;
- mA;
- instance F1;
- weakest source-domain score;
- learned prototype gate statistics.

Prototype learning is worth submitting only if it improves robust validation or
the ensemble calibrator gives it a meaningful contribution.

A private leaderboard score cannot be guaranteed in advance.


## Codabench runtime packaging fix

- `run.py` now loads the ensemble lazily in `load_model()`.
- `BATCH_SIZE = 64` matches the challenge-facing API.
- inference uses an internal micro-batch (default 16) to lower OOM risk.
- `predict_image()` and `predict_batch()` validate the official
  `image_path` / `attribute_names` fields.
- probabilities are converted to finite values and clamped to `[0, 1]`.
- `torch.load` supports environments both with and without `weights_only`.
- zero-weight ensemble members are skipped at runtime.
- the exporter executes the exact created ZIP in a fresh process and calls
  `load_model()`, `predict_image()`, and `predict_batch()`.

Only upload a ZIP when the export log contains:

`SUBMISSION_RUNTIME_CHECK: PASS`
