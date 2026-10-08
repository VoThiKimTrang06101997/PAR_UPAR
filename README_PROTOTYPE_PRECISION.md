# Prototype Precision Recovery

This patch is designed around the observed hidden result:

- Challenge: 0.7438
- mA: 0.7491
- Instance-F1: 0.7386
- Precision: 0.7098
- Recall: 0.7851

The prototype model improved mA strongly but became recall-heavy. The goal of
this patch is to preserve prototype gains while reducing false positives.

## Fast path — no retraining

Use existing:
- prototype_convnext_tiny_seed42_best.pt
- prototype_convnext_tiny_seed123_best.pt

Then run:
1. group-wise prototype-strength calibration;
2. ensemble-weight search;
3. precision-aware LODO threshold calibration;
4. flip-TTA export.

Prototype contribution is calibrated separately for:
- age
- gender
- hair
- upper body
- lower body
- lower-body type
- accessories

A group is allowed to use scale 0, which falls back to the strong linear branch.

## Optional retraining

`train_prototype_precision.py` uses:
- prototype gate init 0.22;
- prototype gate max 0.60;
- prototype auxiliary loss 0.14;
- domain alignment loss 0.012;
- positive-weight cap 2.0;
- one-sided hard-negative loss;
- mild gate-anchor regularization;
- the same EMA, optional KD, LODO validation, and checkpoint logic.

The new checkpoint prefix is:

`prototype_precision_convnext_tiny_seed{seed}_best.pt`

so existing prototype checkpoints are never overwritten.

## Main final submission

`Submission_PROTOTYPE_PRECISION_TTA.zip`

Only upload the ZIP if the exporter prints:

`SUBMISSION_RUNTIME_CHECK: PASS`

A score of 0.76–0.77 is a target, not a guaranteed private-domain result.
