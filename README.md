# PAR_UPAR — VAPOR-PAR for UPAR Challenge 2027

## Overview

This repository implements **VAPOR-PAR** (**Vision-Language Attribute Prototypes with Ontology-aware Domain Robustness**) for **UPAR Challenge 2027 — Track 1: Pedestrian Attribute Recognition**.

The task is a **40-label multi-label pedestrian attribute recognition problem under hidden-domain shift**. The development data are built from **Market1501, PA-100K, and PETA**, while the final evaluation is performed on an unseen private test domain.

The central research question is:

> **How can a pedestrian attribute model learn attribute-specific, label-balanced, semantically grounded representations that remain stable across unseen surveillance domains?**

VAPOR-PAR addresses this by combining a Vision-Language Model with attribute-specific semantic queries, visual-language dual prototypes, domain generalization, ontology-aware supervision, consistency regularization, and calibration.

---

## Core Idea

A conventional pedestrian attribute classifier usually learns:

```text
Image
  ↓
Backbone
  ↓
Global Feature
  ↓
40-way Classification Head
```

This can overfit source-domain shortcuts such as camera style, background, illumination, resolution, clothing distribution, or dataset-specific biases.

VAPOR-PAR instead learns a **structured cross-modal prototype space**:

```text
Image
  ↓
SigLIP2 Vision Encoder
  ↓
Patch Features
  ↓
Semantic Attribute Queries
  ↓
Attribute-specific Cross Attention
  ↓
40 Attribute Features
  ↓
Visual + Language Positive/Negative Dual Prototypes
  ↓
Cosine Prototype Margin
  ↓
40 Attribute Logits
  ↓
Semantic Calibration
  ↓
40 Probabilities
```

Text is not used only as auxiliary input. It acts as a **domain-invariant semantic coordinate system** to which visual representations from different source domains are aligned.

---

# 1. Task Formulation

For each pedestrian image:

```text
x ∈ R^(H × W × 3)
```

the model predicts:

```text
y = [y1, y2, ..., y40]
yi ∈ {0, 1}
```

A pedestrian may simultaneously have multiple attributes, for example:

```text
Adult
Female
Long Hair
White Upper Body
Trousers
Backpack
Glasses
```

Therefore, the problem is **multi-label binary classification**, not single-label classification.

The 40 attributes cover groups such as:

- Age
- Gender
- Hair
- Upper-body length
- Upper-body color
- Lower-body length
- Lower-body color
- Lower-body type
- Accessories

The key challenge is domain generalization:

```text
P_train(X, Y) ≠ P_test(X, Y)
```

The model is trained on source surveillance datasets but must remain robust on an unseen target domain.

---

# 2. VAPOR-PAR Architecture

## 2.1 SigLIP2 Vision-Language Backbone

VAPOR-PAR uses **SigLIP2** as the Vision-Language backbone.

The image encoder produces patch-level visual features:

```text
V = [v1, v2, ..., vM]
V ∈ R^(M × D)
```

The text encoder produces semantic representations for pedestrian attributes.

SigLIP2 is preferred over a plain image classifier because it provides a pretrained cross-modal semantic space that can act as a stable semantic anchor under domain shift.

---

## 2.2 Positive / Negative Semantic Prompt Bank

Each attribute has both positive and negative textual descriptions.

Example for `Accessory-Backpack`:

### Positive prompts

```text
a pedestrian carrying a backpack
a person with a backpack on the back
a surveillance image of a pedestrian wearing a backpack
```

### Negative prompts

```text
a pedestrian without a backpack
a person whose back has no backpack
a pedestrian carrying no backpack
```

For attribute `c`, the text encoder produces:

```text
e(c,k)+ = E_text(t(c,k)+)
e(c,k)- = E_text(t(c,k)-)
```

The positive and negative semantic prototypes are obtained by averaging prompt embeddings:

```text
s(c)+ = Normalize(mean_k(e(c,k)+))
s(c)- = Normalize(mean_k(e(c,k)-))
```

This gives every binary attribute an explicit positive and negative semantic direction.

---

# 3. Domain-Diverse Prompting

A single prompt may itself contain contextual bias.

To reduce this problem, VAPOR-PAR uses multiple domain-conditioned prompts.

Example for `Accessory-Hat`:

```text
a pedestrian wearing a hat
a low-resolution CCTV image of a pedestrian wearing a hat
a nighttime surveillance image of a pedestrian wearing a hat
a blurred surveillance image of a pedestrian wearing a hat
an occluded surveillance image of a pedestrian wearing a hat
a high-angle camera image of a pedestrian wearing a hat
```

The goal is to preserve the semantic identity of an attribute under variations in:

```text
illumination
resolution
blur
viewpoint
occlusion
camera style
surveillance conditions
```

The prompt bank is frozen once generated.

---

# 4. Semantic Attribute Queries

Instead of initializing all 40 attribute queries purely at random, VAPOR-PAR injects language semantics into each query.

For attribute `c`:

```text
semantic_direction(c) =
    Normalize(s(c)+ - s(c)-)
```

The final attribute query is:

```text
q(c) =
    q_learnable(c)
    + W_semantic × semantic_direction(c)
```

This means the query for `Hat`, `Backpack`, `Long Hair`, or `White Upper Body` already carries information about its semantic concept before task-specific training.

---

# 5. Vision-Language Attribute Cross Attention

The model uses 40 semantic queries to retrieve attribute-specific evidence from visual patch features.

Given:

```text
Q = semantic attribute queries
K = visual patch keys
V = visual patch values
```

cross attention is:

```text
Attention(Q, K, V)
    = Softmax(QK^T / sqrt(D)) V
```

The result is:

```text
H = [h1, h2, ..., h40]
H ∈ R^(40 × D)
```

Each `h(c)` represents the visual evidence for one specific attribute.

Examples:

```text
Hair query      → head region
Hat query       → head region
Backpack query  → torso / back region
Shoes query     → lower-body / feet region
```

This reduces feature interference compared with using one global feature for all 40 attributes.

---

# 6. Visual-Language Dual Prototypes

The main VAPOR-PAR contribution is a **dual-prototype representation**.

For every attribute `c`, the model maintains:

### Visual prototypes

```text
p(c)+
p(c)-
```

### Language prototypes

```text
s(c)+
s(c)-
```

A learnable gate controls their contribution:

```text
alpha(c) = sigmoid(a(c))
```

The fused positive prototype is:

```text
r(c)+ =
    Normalize(
        alpha(c) × p(c)+
        + (1 - alpha(c)) × W_text s(c)+
    )
```

The fused negative prototype is:

```text
r(c)- =
    Normalize(
        alpha(c) × p(c)-
        + (1 - alpha(c)) × W_text s(c)-
    )
```

The visual prototype adapts to pedestrian data, while the language prototype acts as a semantic anchor.

---

# 7. Prototype-Margin Classification

VAPOR-PAR does not use a standard fully connected 40-label classifier.

For attribute `c`:

```text
score(c)+ = cosine(h(c), r(c)+)
score(c)- = cosine(h(c), r(c)-)
```

The binary logit is:

```text
z(c) =
    (score(c)+ - score(c)-) / temperature
```

and:

```text
p(c) = sigmoid(z(c))
```

This encourages the model to classify attributes according to semantic prototype directions rather than dataset-specific linear classifier weights.

---

# 8. Balanced Asymmetric Attribute Loss

UPAR attributes are strongly imbalanced.

Let:

```text
pi(c) = positive frequency of attribute c
```

The model uses class-aware positive and negative weights:

```text
w(c)+ = 1 / (2 × pi(c))
w(c)- = 1 / (2 × (1 - pi(c)))
```

Together with asymmetric focusing:

```text
L_bal =
    - mean [
        w+ × y × (1-p)^gamma+ × log(p)
        +
        w- × (1-y) × p^gamma- × log(1-p)
    ]
```

This reduces domination by easy negative samples and improves learning for rare positive attributes.

---

# 9. Semantic Contrastive Supervision

For each attribute-specific visual feature:

```text
h(i,c)
```

the model should be closer to the correct semantic prototype.

For a positive label:

```text
h(i,c) → s(c)+
h(i,c) away from s(c)-
```

A semantic binary contrastive objective is used:

```text
semantic_logit(i,c) =
    cosine(h(i,c), s(c)+)
    - cosine(h(i,c), s(c)-)
```

and optimized using binary cross-entropy with logits.

This gives VLM-based semantic supervision in addition to ground-truth attribute labels.

---

# 10. Cross-Domain Attribute Alignment

The source data come from multiple datasets:

```text
Market1501
PA100K
PETA
```

For an attribute `c`, the model estimates domain-specific feature centroids:

```text
mu(c,d)+
mu(c,d)-
```

where `d` is the source domain.

The domain generalization objective minimizes dispersion between source-domain centroids.

Conceptually:

```text
Female_Market
≈ Female_PA100K
≈ Female_PETA
```

and similarly for negative features.

The purpose is to make the model represent the semantic attribute itself rather than the camera or dataset identity.

---

# 11. Semantic Anchor Loss

Cross-domain alignment alone can still collapse toward a source-specific common feature.

VAPOR-PAR therefore anchors the learned visual prototype space to the language prototype space.

For each attribute:

```text
L_anchor =
    1 - cosine(visual_positive, language_positive)
    +
    1 - cosine(visual_negative, language_negative)
```

The desired structure becomes:

```text
Market visual prototype
≈ PA100K visual prototype
≈ PETA visual prototype
≈ Language semantic prototype
```

---

# 12. Prompt Invariance Loss

Different domain-conditioned prompts should encode the same underlying attribute.

For each prompt variant `k`:

```text
margin(i,c,k) =
    cosine(h(i,c), s(c,k)+)
    -
    cosine(h(i,c), s(c,k)-)
```

VAPOR-PAR minimizes the variance across prompt contexts:

```text
L_prompt =
    mean Var_k(margin(i,c,k))
```

This explicitly encourages language-guided domain invariance.

---

# 13. Semantic Relation Geometry

Simple orthogonality can be inappropriate because real pedestrian attributes may be correlated.

Instead of forcing:

```text
P P^T ≈ I
```

VAPOR-PAR preserves semantic relations from the text space.

Language similarity matrix:

```text
S_text = S S^T
```

Visual prototype similarity matrix:

```text
S_visual = P P^T
```

Relation loss:

```text
L_rel =
    || S_visual - stopgrad(S_text) ||_F^2
```

The visual prototype space therefore learns a geometry compatible with semantic relationships between attributes.

---

# 14. Ontology-Aware Attribute Supervision

The 40 labels are not completely independent.

Examples of structured groups include:

```text
Age:
Young / Adult / Old

Hair:
Short / Long / Bald

Upper-body color:
Black / Blue / Brown / ... / Other

Lower-body color:
Black / Blue / Brown / ... / Other

Lower-body type:
Trousers&Shorts / Skirt&Dress
```

For samples with exactly one valid label in an ontology group, an auxiliary group softmax loss is applied.

This reduces contradictory predictions while respecting unknown or missing annotations.

---

# 15. Weak / Strong Domain Consistency

Each image is transformed into:

```text
x_weak
x_strong
```

The model predicts:

```text
p_weak
p_strong
```

For confident weak predictions, the model minimizes disagreement between weak and strong views.

Prediction consistency:

```text
L_consistency =
    BCEWithLogits(
        strong_logits,
        stopgrad(p_weak)
    )
```

Feature consistency:

```text
L_feature =
    1 - cosine(
        h_weak,
        h_strong
    )
```

This encourages stable attribute representations under appearance perturbations.

---

# 16. Semantic Calibration

Vision-language models may have strong ranking ability but poorly calibrated binary outputs.

VAPOR-PAR therefore applies per-attribute vector scaling:

```text
z_calibrated(c) =
    exp(a(c)) × z(c) + b(c)
```

followed by:

```text
p_calibrated(c) =
    sigmoid(z_calibrated(c))
```

Per-attribute thresholds can also be estimated from development data.

Calibration is particularly important because UPAR evaluation depends on both positive/negative balance and instance-level multi-label prediction quality.

---

# 17. Total Training Objective

The complete VAPOR-PAR objective is:

```text
L_total =
    w_bal      × L_bal
  + w_sem      × L_sem
  + w_dg       × L_domain
  + w_anchor   × L_anchor
  + w_prompt   × L_prompt
  + w_rel      × L_relation
  + w_onto     × L_ontology
  + w_cons     × L_consistency
  + w_feat     × L_feature_consistency
```

A practical starting configuration is:

```text
w_bal     = 1.00
w_sem     = 0.20
w_dg      = 0.10
w_anchor  = 0.10
w_prompt  = 0.05
w_rel     = 0.01
w_onto    = 0.10
w_cons    = 0.20
w_feat    = 0.05
```

These are starting hyperparameters rather than fixed challenge-specific constants.

---

# 18. Training Curriculum

VAPOR-PAR is trained progressively.

## Stage A — Semantic Warm-up

The SigLIP2 backbone is frozen.

Train mainly:

```text
Semantic Queries
Dual Prototype Head
Balanced Attribute Objective
Semantic Contrastive Objective
Semantic Anchor
Ontology / Prompt / Relation losses
```

This allows task-specific components to stabilize before adapting the visual backbone.

## Stage B — Domain Generalization

Enable:

```text
Cross-Domain Alignment
Prompt Invariance
Ontology Supervision
Relation Geometry
Weak / Strong Consistency
```

The purpose is to improve robustness across heterogeneous surveillance domains.

## Stage C — Robust Visual Adaptation

The final vision blocks can be adapted with a very small learning rate.

The goal is to obtain task-specific visual adaptation without destroying the pretrained vision-language representation.

---

# 19. Evaluation

The main validation quantities are:

```text
mean Accuracy (mA)
instance Precision
instance Recall
instance F1
```

A development challenge score can be monitored as:

```text
score =
    (mA + instance_F1) / 2
```

Because the private test domain is unseen, source-domain validation alone should not be treated as a complete estimate of final generalization.

A stronger research protocol is **Leave-One-Domain-Out (LODO)**:

```text
Train: PA100K + PETA
Test : Market1501

Train: Market1501 + PETA
Test : PA100K

Train: Market1501 + PA100K
Test : PETA
```

This better measures whether a method improves domain generalization rather than only in-domain validation.

---

# 20. Recommended Ablation Study

A useful ablation sequence is:

```text
SigLIP2 baseline
+ Semantic Attribute Queries
+ Positive/Negative Semantic Prototypes
+ Visual-Language Dual Prototypes
+ Domain-Diverse Prompting
+ Cross-Domain Alignment
+ Semantic Anchor
+ Prompt Invariance
+ Semantic Relation Geometry
+ Ontology Loss
+ Weak/Strong Consistency
+ Semantic Calibration
VAPOR-PAR Full
```

For each variant, report:

```text
mA
Instance F1
Combined Score
```

and ideally the three LODO evaluations.

---

# 21. Main Research Contributions

VAPOR-PAR is designed around three main contributions.

## 1. Semantic Dual-Prototype Learning

Each pedestrian attribute is represented simultaneously by:

```text
visual positive / negative prototypes
+
VLM-derived language positive / negative prototypes
```

This combines dataset adaptation with semantic grounding.

## 2. Domain-Diverse Semantic Grounding

Multiple surveillance-condition prompts encourage attribute identity to remain stable under:

```text
resolution changes
illumination changes
blur
occlusion
viewpoint
camera variation
```

## 3. Ontology-Aware Cross-Domain Calibration

The method combines:

```text
label-balanced optimization
cross-domain representation alignment
attribute ontology
prediction consistency
semantic calibration
```

to reduce class imbalance, domain drift, and precision-recall instability.

---

# 22. Why VAPOR-PAR Is Different from a Plain VLM Classifier

A plain VLM-based PAR model can be summarized as:

```text
Image
↔
Text
→
Classifier
```

VAPOR-PAR instead learns:

```text
Image
↔
Attribute-specific Visual Feature
↔
Domain-aware Visual Prototype
↔
Language Semantic Prototype
```

Therefore, language is not only an auxiliary feature source.

It becomes a semantic reference system that constrains the structure of the learned visual attribute space.

---

# 23. Repository Structure

```text
PAR_UPAR/
│
├── vapor_par/
│   ├── config.py
│   ├── data.py
│   ├── prompts.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── calibration.py
│   ├── checkpoint.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── submission_builder/
│   ├── build_submission.py
│   └── submission_runtime/
│       ├── run.py
│       └── model_runtime.py
│
├── prepare_data.py
├── train_colab.py
├── inspect_checkpoint.py
├── infer_submit.py
├── requirements.txt
└── README.md
```

---

# 24. Dataset Preparation

The repository is designed for the official UPAR 2027 data organization.

Expected annotations:

```text
UPAR-Challenge-2027/
└── data/
    └── annotations/
        └── task1/
            ├── train/
            │   └── gt.csv
            └── val/
                └── gt.csv
```

The annotation table contains:

```text
image path
+
40 binary pedestrian attributes
```

The project uses the official source-dataset identities to construct domain-aware training signals.

---

# 25. Training

Example:

```bash
python train_colab.py \
    --epochs 25 \
    --batch-size 12 \
    --eval-batch-size 24 \
    --num-workers 2 \
    --save-every-steps 100 \
    --log-every-steps 50
```

The optimizer is AdamW with separate learning rates for the task-specific head and adapted vision backbone.

Typical configuration:

```text
Head LR       : 3e-4
Backbone LR   : 5e-6
Weight decay  : 0.05
Scheduler     : cosine
Gradient clip : 5.0
AMP           : enabled
```

---

# 26. Checkpoints and Reproducibility

Training checkpoints store the state required to continue optimization, including:

```text
VAPOR-PAR trainable parameters
optimizer state
scheduler state
AMP scaler
epoch
next training batch
global step
best validation score
thresholds
calibration
random-number-generator state
```

The pretrained SigLIP2 backbone is loaded from the local cache during training and can be bundled completely for offline challenge inference.

---

# 27. Codabench Inference

For final challenge evaluation, the trained checkpoint is converted into a self-contained offline inference package.

The final archive contains:

```text
run.py
metadata.yaml
model_runtime.py
weights/
vendor/
```

The submission runtime exposes:

```python
load_model()
predict_image(sample)
predict_batch(samples)
```

and returns exactly 40 probabilities in the attribute order requested by the challenge ingestion program.

This is an inference/deployment component of the repository; it is separate from the VAPOR-PAR research method itself.

---

# 28. Research Direction

The intended paper direction is:

> **Domain-Invariant Vision-Language Attribute Prototypes for Generalizable Pedestrian Attribute Recognition**

The main hypothesis is that:

```text
attribute-specific visual representations
+
vision-language semantic prototypes
+
cross-domain alignment
+
label-balanced optimization
```

reduce source-domain shortcuts and attribute entanglement, improving performance on unseen surveillance domains.

---

# 29. Summary

VAPOR-PAR combines:

```text
SigLIP2
+ Domain-Diverse Positive/Negative Prompt Bank
+ Semantic Attribute Queries
+ Attribute-specific Cross Attention
+ Visual-Language Dual Prototypes
+ Prototype-Margin Classification
+ Balanced Asymmetric Learning
+ Semantic Contrastive Supervision
+ Cross-Domain Alignment
+ Semantic Anchor
+ Prompt Invariance
+ Semantic Relation Geometry
+ Ontology-Aware Learning
+ Weak/Strong Consistency
+ Semantic Calibration
```

The central design principle is:

> **Learn a domain-robust, semantically structured prototype space for each pedestrian attribute rather than learning a source-specific 40-label classifier.**
