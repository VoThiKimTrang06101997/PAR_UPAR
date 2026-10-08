from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

NUM_ATTRIBUTES = 40


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
        prototype_gate_min: float = 0.08,
        prototype_gate_max: float = 0.92,
        prototype_ema_momentum: float = 0.95,
        prototype_ema_mix: float = 0.25,
        pretrained: bool = True,
    ):
        super().__init__()

        self.num_attributes = int(num_attributes)
        self.prototype_dim = int(prototype_dim)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_gate_min = float(prototype_gate_min)
        self.prototype_gate_max = float(prototype_gate_max)
        if not (0.0 <= self.prototype_gate_min < self.prototype_gate_max <= 1.0):
            raise ValueError("Invalid prototype gate range")
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
        gate = gate.clamp(
            self.prototype_gate_min,
            self.prototype_gate_max,
        )

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



# Standalone Codabench submission entrypoint. No imports from participant repo.
import gc
import os
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 64
MICRO_BATCH_SIZE = max(1, int(os.environ.get('UPAR_MICRO_BATCH_SIZE', '8')))
_RUNTIME = None


def _load_bundle(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


class StrongTinyForSubmission(nn.Module):
    def __init__(self):
        super().__init__()
        base = convnext_tiny(weights=None)
        ch = int(base.classifier[2].in_features)
        base.classifier[2] = nn.Sequential(nn.Dropout(.22), nn.Linear(ch, 40))
        self.net = base

    def forward(self, x):
        return self.net(x)


class Runtime:
    def __init__(self):
        bundle = _load_bundle(ROOT/'assets'/'model.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attrs = list(bundle['attribute_names'])
        if len(self.attrs) != 40 or len(set(self.attrs)) != 40:
            raise ValueError('Expected 40 unique attribute names')
        self.index = {name:i for i,name in enumerate(self.attrs)}
        self.threshold_logits = torch.as_tensor(bundle['threshold_logits'],dtype=torch.float32).reshape(1,-1).to(self.device)
        if self.threshold_logits.shape != (1,40):
            raise ValueError('Invalid thresholds shape')
        self.strength = float(bundle['strength'])
        self.transform = transforms.Compose([
            transforms.Resize((int(bundle['image_height']),int(bundle['image_width'])),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
        ])
        self.members = []
        for entry in bundle['members']:
            kind = str(entry['kind'])
            state = entry['state_dict']
            if kind in ('prototype','trained'):
                cfg = dict(entry.get('config', {}))
                cfg.pop('pretrained', None)
                model = HybridPrototypePAR(pretrained=False, **cfg)
            elif kind == 'strong':
                model = StrongTinyForSubmission()
            else:
                raise ValueError('Unknown model type '+kind)
            model.load_state_dict(state, strict=True)
            entry['state_dict'] = None
            del state
            model = model.to(self.device).eval()
            self.members.append((kind, float(entry['weight']), model))
        del bundle
        gc.collect()
        if not self.members:
            raise ValueError('No active model members')
        if abs(sum(w for _,w,_ in self.members)-1.)>1e-3:
            raise ValueError('Ensemble weights must sum to one')
        if self.device.type=='cuda':
            torch.cuda.empty_cache()

    def _image(self, name):
        with Image.open(name) as im:
            return self.transform(im.convert('RGB'))

    def _logits(self, kind, model, x):
        if kind == 'strong':
            return model(x)
        info = model(x, return_aux=True)
        gate = torch.clamp(info['prototype_gate'] * self.strength, 0., .60)
        return (1.-gate)*info['linear_logits'] + gate*info['proto_logits']

    @torch.inference_mode()
    def predict(self, samples):
        predictions=[]
        for start in range(0,len(samples),MICRO_BATCH_SIZE):
            piece = samples[start:start+MICRO_BATCH_SIZE]
            x=torch.stack([self._image(s['image_path']) for s in piece]).to(self.device)
            z=None
            for kind, weight, model in self.members:
                if self.device.type=='cuda':
                    with torch.autocast('cuda',dtype=torch.float16):
                        a=self._logits(kind,model,x)
                        b=self._logits(kind,model,torch.flip(x,dims=[3]))
                else:
                    a=self._logits(kind,model,x)
                    b=self._logits(kind,model,torch.flip(x,dims=[3]))
                zz=.5*(a.float()+b.float())*weight
                z = zz if z is None else z + zz
            probs=torch.nan_to_num(torch.sigmoid(z-self.threshold_logits),nan=.5,posinf=1.,neginf=0.).clamp(0.,1.).cpu()
            for j,s in enumerate(piece):
                names=list(s['attribute_names'])
                if len(names)!=40 or len(set(names))!=40:
                    raise ValueError('Invalid number of requested attribute_names')
                indices=[self.index[name] for name in names]
                result=probs[j,indices].tolist()
                if not all(math.isfinite(float(v)) and 0.<=float(v)<=1. for v in result):
                    raise ValueError('Invalid probability')
                predictions.append([float(v) for v in result])
            del x, probs, z
        return predictions


def _ensure():
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME=Runtime()
    return _RUNTIME


def load_model():
    """Codabench load step."""
    _ensure()
    return None


def predict_image(sample):
    return _ensure().predict([sample])[0]


def predict_batch(samples):
    if not samples:
        return []
    return _ensure().predict(list(samples))
