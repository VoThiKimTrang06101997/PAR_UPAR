from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
)

ATTRIBUTE_NAMES = [
    "Age-Young","Age-Adult","Age-Old","Gender-Female",
    "Hair-Length-Short","Hair-Length-Long","Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black","UpperBody-Color-Blue","UpperBody-Color-Brown",
    "UpperBody-Color-Green","UpperBody-Color-Grey","UpperBody-Color-Orange",
    "UpperBody-Color-Pink","UpperBody-Color-Purple","UpperBody-Color-Red",
    "UpperBody-Color-White","UpperBody-Color-Yellow","UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black","LowerBody-Color-Blue","LowerBody-Color-Brown",
    "LowerBody-Color-Green","LowerBody-Color-Grey","LowerBody-Color-Orange",
    "LowerBody-Color-Pink","LowerBody-Color-Purple","LowerBody-Color-Red",
    "LowerBody-Color-White","LowerBody-Color-Yellow","LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts","LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack","Accessory-Bag","Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun","Accessory-Hat",
]
NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)
DOMAIN_NAMES = ["Market1501", "PA100k", "PETA"]

# Low-weight structural prior. These groups are mostly mutually exclusive in UPAR.
ATTRIBUTE_GROUPS = [
    (0, 1, 2),       # age
    (4, 5, 6),       # hair length/state
    tuple(range(8, 20)),   # upper-body color
    tuple(range(21, 33)),  # lower-body color
    (33, 34),        # lower-body type
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_domain_id(path: str) -> int:
    s = str(path).replace("\\", "/").lower()
    if "market1501" in s or "/market" in s:
        return 0
    if "pa100k" in s or "pa-100k" in s:
        return 1
    if "peta" in s:
        return 2
    return -1


def detect_image_column(df):
    for candidate in ("# image", "image", "image_path", "path", "filename"):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        vals = df[col].astype(str).head(100).str.lower()
        if vals.str.endswith((".jpg", ".jpeg", ".png", ".bmp")).any():
            return col
    raise RuntimeError("Cannot detect image path column")


class ImageResolver:
    def __init__(self, data_root: str | Path, repo_root: str | Path):
        self.data_root = Path(data_root)
        self.repo_root = Path(repo_root)
        self._index = None

    def _build_index(self):
        if self._index is not None:
            return
        self._index = {}
        for p in self.data_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                self._index.setdefault(p.name, p)

    def resolve(self, value) -> Path:
        raw = Path(str(value))
        for p in (raw, self.data_root / raw, self.repo_root / raw):
            if p.exists():
                return p
        self._build_index()
        match = self._index.get(raw.name)
        if match is None:
            raise FileNotFoundError(f"Cannot resolve image: {value}")
        return match


def build_train_transform(height=288, width=144):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            (height, width), scale=(0.74, 1.0), ratio=(0.40, 0.60),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18, hue=0.04)
        ], p=0.7),
        transforms.RandomGrayscale(p=0.04),
        transforms.RandomApply([
            transforms.GaussianBlur(3, sigma=(0.1, 1.0))
        ], p=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.22, scale=(0.02,0.14), ratio=(0.35,2.8), value="random"),
    ])


def build_eval_transform(height=288, width=144):
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])


class UPARDataset(Dataset):
    def __init__(self, dataframe, resolver: ImageResolver, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_col = detect_image_column(self.df)
        self.paths = self.df[self.image_col].astype(str).tolist()
        self.targets = self.df[ATTRIBUTE_NAMES].to_numpy(dtype=np.float32)
        self.domains = np.asarray([infer_domain_id(p) for p in self.paths], dtype=np.int64)
        self.resolver = resolver
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.resolver.resolve(self.paths[idx])
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        labels = self.targets[idx].copy()
        valid = ((labels == 0) | (labels == 1)).astype(np.float32)
        labels = np.where(valid > 0, labels, 0).astype(np.float32)
        return {
            "image": image,
            "target": torch.from_numpy(labels),
            "valid": torch.from_numpy(valid),
            "domain": int(self.domains[idx]),
            "path": self.paths[idx],
        }


def make_domain_balanced_sampler(domains: np.ndarray, seed=42):
    domains = np.asarray(domains, dtype=np.int64)
    weights = np.ones(len(domains), dtype=np.float64)
    for d in sorted(set(int(x) for x in domains.tolist() if int(x) >= 0)):
        n = int((domains == d).sum())
        if n:
            weights[domains == d] = 1.0 / n
    gen = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), True, generator=gen)


class MixStyle(nn.Module):
    def __init__(self, p=0.35, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = float(p), float(alpha), float(eps)

    def forward(self, x):
        if (not self.training) or x.ndim != 4 or x.size(0) < 2:
            return x
        if torch.rand((), device=x.device).item() > self.p:
            return x
        mu = x.mean((2,3), keepdim=True)
        sig = (x.var((2,3), keepdim=True, unbiased=False) + self.eps).sqrt()
        mu_d, sig_d = mu.detach(), sig.detach()
        x_n = (x - mu_d) / sig_d
        perm = torch.randperm(x.size(0), device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0),1,1,1)).to(x.device, x.dtype)
        return x_n * (lam*sig_d + (1-lam)*sig_d[perm]) + (lam*mu_d + (1-lam)*mu_d[perm])


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    return _GRL.apply(x, lambd)


class ConvNeXtPAR(nn.Module):
    """ConvNeXt PAR model. ConvNeXt uses LayerNorm rather than BatchNorm, useful under domain shift."""
    def __init__(self, backbone="tiny", num_attributes=NUM_ATTRIBUTES, pretrained=True, mixstyle=True):
        super().__init__()
        if backbone == "tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            try:
                base = convnext_tiny(weights=weights)
            except Exception:
                base = convnext_tiny(weights=None)
        elif backbone == "small":
            weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            try:
                base = convnext_small(weights=weights)
            except Exception:
                base = convnext_small(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = copy.deepcopy(base.classifier[0])
        in_features = base.classifier[2].in_features
        self.feature_dim = int(in_features)
        self.dropout = nn.Dropout(0.25 if backbone == "tiny" else 0.30)
        self.attr_head = nn.Linear(in_features, num_attributes)
        self.domain_head = nn.Sequential(
            nn.Linear(in_features, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 3)
        )
        self.mixstyle = bool(mixstyle)
        self.mix1 = MixStyle(p=0.35, alpha=0.1)
        self.mix2 = MixStyle(p=0.25, alpha=0.1)

    def extract_features(self, x):
        for idx, block in enumerate(self.features):
            x = block(x)
            if self.mixstyle and idx == 3:
                x = self.mix1(x)
            elif self.mixstyle and idx == 5:
                x = self.mix2(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return torch.flatten(x, 1)

    def forward(self, x, grl_lambda=0.0, return_domain=False, return_features=False):
        feat = self.extract_features(x)
        logits = self.attr_head(self.dropout(feat))
        if return_domain:
            dlogits = self.domain_head(grad_reverse(feat, grl_lambda))
            return logits, dlogits, feat
        if return_features:
            return logits, feat
        return logits


class MaskedAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, valid):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss += (1.0-targets) * torch.log(xs_neg.clamp(min=self.eps))
        with torch.no_grad():
            pt = xs_pos*targets + xs_neg*(1.0-targets)
            gamma = self.gamma_pos*targets + self.gamma_neg*(1.0-targets)
        loss *= torch.pow(1.0-pt, gamma)
        loss = -loss * valid
        return loss.sum() / valid.sum().clamp(min=1.0)


def domain_loss(logits, domains):
    domains = domains.long()
    valid = domains >= 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], domains[valid])


def confidence_distillation_loss(student_logits, teacher_logits, valid, temperature=2.0, unknown_weight=0.15):
    t = float(temperature)
    teacher_prob = torch.sigmoid(teacher_logits.detach() / t)
    bce = F.binary_cross_entropy_with_logits(student_logits / t, teacher_prob, reduction="none") * (t*t)
    confidence = (2.0 * (teacher_prob - 0.5).abs()).pow(1.5).detach()
    mask = valid + float(unknown_weight) * (1.0 - valid)
    w = confidence * mask
    return (bce * w).sum() / w.sum().clamp(min=1.0)


def feature_distillation_loss(student_feat, teacher_feat):
    if student_feat.shape[1] != teacher_feat.shape[1]:
        # Tiny and Small currently share the same final width in torchvision,
        # but keep a safe dimensional crop for forward compatibility.
        d = min(student_feat.shape[1], teacher_feat.shape[1])
        student_feat = student_feat[:, :d]
        teacher_feat = teacher_feat[:, :d]
    s = F.normalize(student_feat, dim=1)
    t = F.normalize(teacher_feat.detach(), dim=1)
    return (1.0 - (s*t).sum(dim=1)).mean()


def exclusivity_regularizer(logits):
    p = torch.sigmoid(logits)
    vals = []
    for g in ATTRIBUTE_GROUPS:
        q = p[:, list(g)].sum(dim=1)
        vals.append(((q - 1.0) ** 2).mean())
    return torch.stack(vals).mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.9997):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema = self.module.state_dict()
        cur = model.state_dict()
        for k, v in ema.items():
            src = cur[k].detach()
            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(src, alpha=1.0-self.decay)
            else:
                v.copy_(src)


@dataclass
class TrainConfig:
    image_height: int = 288
    image_width: int = 144
    batch_size: int = 48
    num_workers: int = 4
    teacher_epochs: int = 8
    student_epochs: int = 10
    lr_teacher: float = 1.8e-4
    lr_student: float = 2.2e-4
    weight_decay: float = 0.04
    domain_loss_weight: float = 0.035
    kd_weight: float = 0.55
    feature_kd_weight: float = 0.08
    group_weight: float = 0.012
    temperature: float = 2.0
    ema_decay: float = 0.9997
