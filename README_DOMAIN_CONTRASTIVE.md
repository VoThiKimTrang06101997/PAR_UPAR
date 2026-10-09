# UPAR Domain-Contrastive Prototype Training

The uploaded `generalization_audit_report.json` selected the unchanged original
Prototype ensemble; the previous retrained model was rejected. Source-domain
Challenge scores were Market1501 0.8552, PA100K 0.8470, PETA 0.8899.

## What changed

1. **Fresh student** (ImageNet ConvNeXt-Tiny initialization, new attribute
   queries, new learned + EMA positive/negative prototype bank). The old
   Prototype checkpoint is NOT copied into the student weights.
2. **Masked Balanced BCE** is retained as the primary classification loss.
3. **Label-masked cross-domain supervised contrastive loss**: same-attribute,
   same-label samples from DIFFERENT source domains are positives;
   opposite-label samples provide negatives. Missing labels do not enter it.
4. **Positive/negative cross-domain prototype centroid alignment** remains a
   separate small regularizer. Balanced mini-batches make cross-domain pairs
   available more consistently than a global random weighted sampler.
5. **Teacher only in final all-source training**: old Prototype model remains
   frozen for modest output distillation. No full-source teacher in LODO folds.
6. **TRUE LODO hyperparameter selection**: each candidate runs three separate
   training experiments, each trained on 2 source datasets and validated on
   the third. No held-out-domain images are included in the fold's training
   sampler. Fold scores use a fixed 0.5 operating threshold, not a fitted
   threshold on the held-out labels. Candidates include zero-loss control.
7. **Final training** re-trains seed 42 and 123 on all three training source
   datasets using LODO-selected loss weights, with teacher regularization.
8. **Final selector** compares original Prototype vs fresh trained student on a
   FIT/AUDIT partition, and exporter refuses to submit retrained weights if the
   source-audit guard rejected them. ZIP includes actual post-export smoke test.

## Source files

- `domain_contrastive.py` (NEW): DomainMixBatchSampler, cross-domain SupCon,
  label-conditioned centroid alignment.
- `train_domain_contrastive.py` (NEW): strict LODO training + final all-source
  student training, EMA, resume, checkpoints and validation.
- `select_domain_objective.py` (NEW): LODO hyperparameter aggregation and
  threshold-free comparison to a zero-contrastive baseline.
- `calibrate_generalization.py` (MODIFIED): now points trained candidate
  filenames at `prototype_contrastive_convnext_tiny_seed*_best.pt`.
- `prototype_par.py`, `export_generalization.py`, `submission_generalization.py`:
  matched, compatible HybridPrototype architecture and Codabench runtime.

The notebook supplies dependency source files (`strong_par.py`,
`train_strong_par.py`, `vapor_par/metrics.py`) only when missing from GitHub.

## Output files (no numbered version suffixes)

- `Checkpoints/prototype_contrastive_convnext_tiny_seed42_best.pt`
- `Checkpoints/prototype_contrastive_convnext_tiny_seed123_best.pt`
- `Results/domain_objective_lodo.json`
- `Results/generalization_audit_report.json`
- `Results/Submission_PROTOTYPE_CONTRASTIVE_TTA.zip` (ONLY IF accepted)

**Training cost:** The strict LODO study includes 3 objective candidates x 3
held-out domains = 9 separate short training runs, then 2 full training runs.
Defaults in the notebook can require many GPU hours; keep Drive checkpoints to
resume. LODO source transfer is not a guarantee for the hidden test.

**Validation policy:** Never use Codabench private outcomes to fit thresholds or
model weights. Holdout results are used to choose loss strength only. The
final source validation is used to fit ensemble thresholds and run the audit.
Do not claim an accepted public or private score before actual evaluation.
