from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from strong_par import NUM_ATTRIBUTES


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


class HybridPrototypePAR(nn.Module):
    """
    ConvNeXt-Tiny + attribute-specific spatial queries + positive/negative prototypes.

    The final logit is a learnable hybrid of:
        1) a conventional global linear classifier, and
        2) a prototype-margin classifier.

    This is intentionally conservative for leaderboard use: if prototypes are
    not yet useful for a specific attribute, the learned gate can keep more
    weight on the linear path instead of forcing pure prototype classification.
    """

    def __init__(
        self,
        num_attributes: int = NUM_ATTRIBUTES,
        prototype_dim: int = 256,
        prototype_temperature: float = 0.20,
        prototype_gate_init: float = 0.35,
        prototype_ema_momentum: float = 0.95,
        prototype_ema_mix: float = 0.25,
        pretrained: bool = True,
    ):
        super().__init__()

        self.num_attributes = int(num_attributes)
        self.prototype_dim = int(prototype_dim)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_ema_momentum = float(prototype_ema_momentum)
        self.prototype_ema_mix = float(prototype_ema_mix)

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            base = convnext_tiny(weights=weights)
        except Exception:
            base = convnext_tiny(weights=None)

        self.backbone = base
        self.backbone_dim = int(base.classifier[2].in_features)

        # We use the pretrained ConvNeXt feature extractor directly.
        # The original 1000-way classifier is not used.
        self.backbone.classifier[2] = nn.Identity()

        # Global linear safety path.
        self.global_dropout = nn.Dropout(0.22)
        self.linear_head = nn.Linear(self.backbone_dim, self.num_attributes)

        # Attribute-specific local path.
        self.token_norm = nn.LayerNorm(self.backbone_dim)
        self.token_proj = nn.Linear(self.backbone_dim, self.prototype_dim, bias=False)
        nn.init.orthogonal_(self.token_proj.weight)
        self.attribute_queries = nn.Parameter(
            torch.empty(self.num_attributes, self.prototype_dim)
        )
        nn.init.trunc_normal_(self.attribute_queries, std=0.02)

        self.attribute_norm = nn.LayerNorm(self.prototype_dim)

        # Learnable positive / negative prototypes.
        self.prototype_pos = nn.Parameter(
            torch.empty(self.num_attributes, self.prototype_dim)
        )
        self.prototype_neg = nn.Parameter(
            torch.empty(self.num_attributes, self.prototype_dim)
        )
        nn.init.trunc_normal_(self.prototype_pos, std=0.02)
        nn.init.trunc_normal_(self.prototype_neg, std=0.02)

        # Stable EMA prototypes updated from labeled batch features.
        with torch.no_grad():
            pos0 = F.normalize(self.prototype_pos.detach().clone(), dim=-1)
            neg0 = F.normalize(self.prototype_neg.detach().clone(), dim=-1)

        self.register_buffer("prototype_pos_ema", pos0)
        self.register_buffer("prototype_neg_ema", neg0)
        self.register_buffer(
            "prototype_pos_updates",
            torch.zeros(self.num_attributes, dtype=torch.long),
        )
        self.register_buffer(
            "prototype_neg_updates",
            torch.zeros(self.num_attributes, dtype=torch.long),
        )

        # Attribute-wise gate. Initial value 0.35 means the model starts closer
        # to the already reliable linear classifier and earns prototype weight
        # during training.
        self.prototype_gate_logits = nn.Parameter(
            torch.full(
                (self.num_attributes,),
                _logit(prototype_gate_init),
                dtype=torch.float32,
            )
        )

    def _backbone_features(self, x: torch.Tensor):
        fmap = self.backbone.features(x)  # [B, C, H, W]

        # Match torchvision ConvNeXt's original global path exactly:
        # features -> avgpool -> classifier LayerNorm2d -> flatten.
        pooled = self.backbone.avgpool(fmap)
        pooled = self.backbone.classifier[0](pooled)
        global_feature = pooled.flatten(1)

        # Spatial tokens for attribute queries.
        tokens = fmap.permute(0, 2, 3, 1).contiguous()
        tokens = self.token_norm(tokens)
        tokens = self.token_proj(tokens)
        tokens = tokens.flatten(1, 2)  # [B, HW, D]
        return global_feature, tokens

    def effective_prototypes(self):
        learned_pos = F.normalize(self.prototype_pos, dim=-1)
        learned_neg = F.normalize(self.prototype_neg, dim=-1)
        ema_pos = F.normalize(self.prototype_pos_ema, dim=-1)
        ema_neg = F.normalize(self.prototype_neg_ema, dim=-1)

        mix = float(self.prototype_ema_mix)
        pos = F.normalize((1.0 - mix) * learned_pos + mix * ema_pos, dim=-1)
        neg = F.normalize((1.0 - mix) * learned_neg + mix * ema_neg, dim=-1)
        return pos, neg

    def _attribute_features(self, tokens: torch.Tensor):
        q = F.normalize(self.attribute_queries, dim=-1)
        k = F.normalize(tokens, dim=-1)

        # [B, C, HW].  Normalized q/k keeps attention numerically stable.
        attn_logits = torch.einsum("bnd,cd->bcn", k, q)
        # q/k are normalized; a small cosine-attention temperature produces
        # meaningful spatial selectivity. Dividing by sqrt(D) here would make
        # the distribution almost uniform.
        attn_logits = attn_logits / 0.10
        attn = torch.softmax(attn_logits, dim=-1)

        features = torch.einsum("bcn,bnd->bcd", attn, tokens)
        features = self.attribute_norm(features)
        features = F.normalize(features, dim=-1)
        return features, attn

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        global_feature, tokens = self._backbone_features(x)

        linear_logits = self.linear_head(
            self.global_dropout(global_feature)
        )

        attr_features, attn = self._attribute_features(tokens)
        pos, neg = self.effective_prototypes()

        pos_sim = torch.einsum("bcd,cd->bc", attr_features, pos)
        neg_sim = torch.einsum("bcd,cd->bc", attr_features, neg)

        proto_logits = (
            pos_sim - neg_sim
        ) / max(float(self.prototype_temperature), 1e-4)

        gate = torch.sigmoid(self.prototype_gate_logits).reshape(1, -1)
        # Keep either branch from disappearing entirely during training.
        gate = gate.clamp(0.08, 0.92)

        logits = (1.0 - gate) * linear_logits + gate * proto_logits

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "linear_logits": linear_logits,
            "proto_logits": proto_logits,
            "features": attr_features,
            "attention": attn,
            "prototype_pos": pos,
            "prototype_neg": neg,
            "prototype_gate": gate,
        }

    @torch.no_grad()
    def update_prototype_ema(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ):
        """
        Update class-conditional prototypes from attribute-specific features.

        Unknown labels are excluded through `valid`.
        """
        features = F.normalize(features.detach().float(), dim=-1)
        targets = targets.detach()
        valid = valid.detach().bool()
        momentum = float(self.prototype_ema_momentum)

        for c in range(self.num_attributes):
            is_valid = valid[:, c]

            pos_mask = is_valid & (targets[:, c] > 0.5)
            if bool(pos_mask.any()):
                center = F.normalize(
                    features[pos_mask, c].mean(dim=0),
                    dim=0,
                )
                if int(self.prototype_pos_updates[c].item()) == 0:
                    self.prototype_pos_ema[c].copy_(center)
                else:
                    mixed = (
                        momentum * self.prototype_pos_ema[c]
                        + (1.0 - momentum) * center
                    )
                    self.prototype_pos_ema[c].copy_(
                        F.normalize(mixed, dim=0)
                    )
                self.prototype_pos_updates[c].add_(1)

            neg_mask = is_valid & (targets[:, c] <= 0.5)
            if bool(neg_mask.any()):
                center = F.normalize(
                    features[neg_mask, c].mean(dim=0),
                    dim=0,
                )
                if int(self.prototype_neg_updates[c].item()) == 0:
                    self.prototype_neg_ema[c].copy_(center)
                else:
                    mixed = (
                        momentum * self.prototype_neg_ema[c]
                        + (1.0 - momentum) * center
                    )
                    self.prototype_neg_ema[c].copy_(
                        F.normalize(mixed, dim=0)
                    )
                self.prototype_neg_updates[c].add_(1)

    @torch.no_grad()
    def load_strong_checkpoint(self, checkpoint: str | Path):
        """
        Warm-start from the existing StrongPARModel checkpoint.

        ConvNeXt feature weights are copied from `net.features.*` and the
        existing global 40-label linear classifier is copied when shapes match.
        Prototype-specific parameters remain newly initialized.
        """
        checkpoint = Path(checkpoint)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)

        state = (
            ck.get("inference_model_state")
            or ck.get("ema_model_state")
            or ck.get("model_state")
            or ck
        )

        own = self.state_dict()
        copied = []
        skipped = []

        for key, value in state.items():
            new_key = None

            if key.startswith("net.features."):
                new_key = "backbone.features." + key[len("net.features."):]
            elif key.startswith("net.classifier.0."):
                new_key = "backbone.classifier.0." + key[len("net.classifier.0."):]
            elif key == "net.classifier.2.1.weight":
                new_key = "linear_head.weight"
            elif key == "net.classifier.2.1.bias":
                new_key = "linear_head.bias"

            if (
                new_key is not None
                and new_key in own
                and tuple(own[new_key].shape) == tuple(value.shape)
            ):
                own[new_key].copy_(value)
                copied.append((key, new_key))
            else:
                skipped.append(key)

        self.load_state_dict(own, strict=True)

        # Semantic warm-start for the prototype branch from the already trained
        # 40-label linear classifier.  The orthogonal random projection largely
        # preserves pairwise classifier geometry while moving it into the
        # prototype space.
        projected = F.linear(
            self.linear_head.weight.detach(),
            self.token_proj.weight.detach(),
        )
        projected = F.normalize(projected, dim=-1)
        self.attribute_queries.copy_(projected)
        self.prototype_pos.copy_(projected)
        self.prototype_neg.copy_(-projected)
        self.prototype_pos_ema.copy_(projected)
        self.prototype_neg_ema.copy_(-projected)
        self.prototype_pos_updates.zero_()
        self.prototype_neg_updates.zero_()

        return {
            "checkpoint": str(checkpoint),
            "copied": len(copied),
            "skipped": len(skipped),
            "prototype_init": "projected_linear_classifier",
            "copied_examples": copied[:12],
        }


def prototype_margin_aux_loss(
    proto_logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
):
    """
    Small auxiliary BCE on the pure prototype branch.
    This forces the prototype head to remain predictive even though the
    deployment logit is a hybrid linear/prototype output.
    """
    raw = F.binary_cross_entropy_with_logits(
        proto_logits,
        targets.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    w = valid.float()
    return (raw * w).sum() / w.sum().clamp(min=1.0)


def prototype_pair_separation_loss(
    prototype_pos: torch.Tensor,
    prototype_neg: torch.Tensor,
    max_cosine: float = 0.20,
):
    """
    Positive and negative prototypes for the same attribute should not collapse.
    Only excessive cosine similarity is penalized; we do not force them to be
    fully opposite because real visual manifolds are not perfectly symmetric.
    """
    sim = (prototype_pos * prototype_neg).sum(dim=-1)
    return F.relu(sim - float(max_cosine)).mean()


def cross_domain_prototype_alignment_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    domains: torch.Tensor,
    min_samples: int = 2,
):
    """
    Align positive/negative attribute centroids across Market1501, PA100K, PETA.

    The loss is computed on normalized attribute-specific features.  Only
    domain/class cells with at least `min_samples` examples contribute.
    """
    features = F.normalize(features, dim=-1)
    targets = targets.float()
    valid = valid.bool()
    domains = domains.long()

    unique_domains = [
        int(d)
        for d in domains.unique().tolist()
        if int(d) >= 0
    ]
    if len(unique_domains) < 2:
        return features.sum() * 0.0

    losses = []
    num_attributes = features.shape[1]

    for c in range(num_attributes):
        for positive in (True, False):
            centers = []

            for d in unique_domains:
                mask = (
                    (domains == d)
                    & valid[:, c]
                    & (
                        (targets[:, c] > 0.5)
                        if positive
                        else (targets[:, c] <= 0.5)
                    )
                )

                if int(mask.sum().item()) < int(min_samples):
                    continue

                center = F.normalize(
                    features[mask, c].mean(dim=0),
                    dim=0,
                )
                centers.append(center)

            if len(centers) >= 2:
                centers = torch.stack(centers, dim=0)
                global_center = F.normalize(
                    centers.mean(dim=0),
                    dim=0,
                )
                # Cosine distance to a shared semantic direction.
                losses.append(
                    1.0
                    - (centers * global_center.unsqueeze(0)).sum(dim=-1).mean()
                )

    if not losses:
        return features.sum() * 0.0

    return torch.stack(losses).mean()
