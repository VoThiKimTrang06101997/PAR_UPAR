"""Domain-aware mini-batches and label-masked, cross-domain supervised contrastive loss.

Only known (binary) attribute annotations are used. A positive pair must share
both the attribute and its 0/1 label while coming from DIFFERENT source domains.
Opposite-label features provide negatives. Same-domain same-label features are
not inadvertently treated as negatives.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class DomainMixBatchSampler(Sampler[list[int]]):
    """Sample each batch from multiple known source domains.

    Outputs positions in a domain-filtered torch.utils.data.Subset. Keeping
    at least a few samples per domain avoids empty cross-domain objectives.
    Sampling is with replacement. Seed + epoch make reruns reproducible.
    """

    def __init__(self, domains, batch_size=48, steps_per_epoch=900,
                 alpha=0.35, seed=42, min_per_domain=8):
        self.domains = np.asarray(domains, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.min_per_domain = int(min_per_domain)
        self.epoch = 0
        self.pools = {
            int(d): np.flatnonzero(self.domains == int(d))
            for d in np.unique(self.domains) if int(d) >= 0
        }
        if len(self.pools) < 2:
            raise ValueError('Cross-domain contrastive training needs >=2 source domains')
        if self.steps_per_epoch < 1:
            raise ValueError('steps_per_epoch must be >= 1')
        if self.batch_size < 2 * len(self.pools):
            raise ValueError('batch_size too small for mixed-domain batches')
        if not (0 <= self.alpha <= 1):
            raise ValueError('alpha must be in [0,1]')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 10007)
        ids = list(self.pools)
        sizes = np.asarray([len(self.pools[d]) for d in ids], dtype=np.float64)
        # Fractional balancing: exponent=1-alpha; alpha=0 is original frequency.
        probs = np.power(sizes, 1.0 - self.alpha)
        probs /= probs.sum()
        reserved = min(self.min_per_domain, self.batch_size // len(ids))
        reserved = max(2, reserved)
        assert reserved * len(ids) <= self.batch_size
        for _ in range(self.steps_per_epoch):
            counts = np.full(len(ids), reserved, dtype=np.int64)
            remain = self.batch_size - int(counts.sum())
            if remain:
                counts += rng.multinomial(remain, probs)
            batch = np.concatenate([
                rng.choice(self.pools[d], size=int(n), replace=True)
                for d, n in zip(ids, counts)
            ]).tolist()
            rng.shuffle(batch)
            yield batch


def cross_domain_supcon(features: torch.Tensor, targets: torch.Tensor,
                        valid: torch.Tensor, domains: torch.Tensor,
                        temperature: float = 0.18, max_per_cell: int = 8,
                        max_attributes: int = 16):
    """Cross-domain, class-conditional InfoNCE on [batch,40,dim] features.

    For each attribute, anchor positives are same-label examples from other
    source domains; negatives have the opposite label (from any domain).
    All terms use valid annotations; attributes without both classes and
    sufficient cross-domain positives are skipped. No detached feature path.
    """
    if features.ndim != 3 or targets.shape != features.shape[:2]:
        raise ValueError('Expected features [B,C,D], targets [B,C]')
    z = F.normalize(features.float(), dim=-1)
    y = targets.float()
    v = valid.bool()
    d = domains.long()
    loss_terms = []
    chosen = torch.randperm(z.shape[1], device=z.device)[:max_attributes].tolist()
    for c in chosen:
        # Subsample each domain/class cell for bounded compute and to prevent
        # PA100K / majority class dominating the contrastive objective.
        keep_parts = []
        for domain in torch.unique(d).tolist():
            if int(domain) < 0:
                continue
            for cls in (0, 1):
                ix = torch.where((d == domain) & v[:, c] & (y[:, c] == cls))[0]
                if ix.numel() > max_per_cell:
                    ix = ix[torch.randperm(ix.numel(), device=ix.device)[:max_per_cell]]
                if ix.numel():
                    keep_parts.append(ix)
        if len(keep_parts) < 3:
            continue
        ix = torch.cat(keep_parts)
        if ix.numel() < 3:
            continue
        zc, yc, dc = z[ix, c], y[ix, c], d[ix]
        same_label = yc[:, None] == yc[None, :]
        different_domain = dc[:, None] != dc[None, :]
        positive = same_label & different_domain
        negative = ~same_label
        eligible = positive | negative
        anchors = positive.any(1) & negative.any(1)
        if not anchors.any():
            continue
        sim = (zc @ zc.T) / max(float(temperature), 1e-4)
        # Log-sum-exp denominator contains only cross-domain positives and
        # opposite-label negatives, not ambiguous same-domain positives.
        denominator = torch.logsumexp(sim.masked_fill(~eligible, -1e4), dim=1)
        numerator = torch.logsumexp(sim.masked_fill(~positive, -1e4), dim=1)
        loss_terms.append((denominator[anchors] - numerator[anchors]).mean())
    if not loss_terms:
        return features.sum() * 0.0
    return torch.stack(loss_terms).mean()


def domain_positive_negative_alignment(features, targets, valid, domains,
                                       min_per_cell=2, max_attributes=16):
    """Align positive with positive and negative with negative across domains.

    A cell must have at least `min_per_cell` valid examples; compare each
    centroid to the balanced mean of domain centroids of the same class.
    """
    z = F.normalize(features.float(), dim=-1)
    d = domains.long()
    y = targets.float()
    v = valid.bool()
    ids = torch.unique(d[d >= 0]).tolist()
    if len(ids) < 2:
        return features.sum() * 0.0
    terms = []
    chosen = torch.randperm(z.shape[1], device=z.device)[:max_attributes].tolist()
    for c in chosen:
        for cls in (0, 1):
            centers = []
            for domain in ids:
                mask = (d == domain) & v[:, c] & (y[:, c] == cls)
                if mask.sum().item() >= min_per_cell:
                    centers.append(F.normalize(z[mask, c].mean(0), dim=0))
            if len(centers) > 1:
                mat = torch.stack(centers)
                centroid = F.normalize(mat.mean(0), dim=0)
                terms.append((1.0 - mat @ centroid).mean())
    return torch.stack(terms).mean() if terms else features.sum() * 0.0
