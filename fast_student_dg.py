from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


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
DOMAIN_TO_ID = {name: idx for idx, name in enumerate(DOMAIN_NAMES)}


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
        values = df[col].astype(str).head(100).str.lower()
        if values.str.endswith((".jpg", ".jpeg", ".png", ".bmp")).any():
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


def build_train_transform(height: int = 256, width: int = 128):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            (height, width),
            scale=(0.72, 1.0),
            ratio=(0.38, 0.58),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.30,
                contrast=0.30,
                saturation=0.25,
                hue=0.06,
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
        ], p=0.15),
        transforms.RandomAutocontrast(p=0.12),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.18),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])


def build_eval_transform(height: int = 256, width: int = 128):
    return transforms.Compose([
        transforms.Resize(
            (height, width),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class UPARFastDataset(Dataset):
    def __init__(self, dataframe, resolver: ImageResolver, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_col = detect_image_column(self.df)
        self.resolver = resolver
        self.transform = transform
        self.paths = self.df[self.image_col].astype(str).tolist()
        self.targets = self.df[ATTRIBUTE_NAMES].to_numpy(dtype=np.float32)
        self.domains = np.asarray([infer_domain_id(p) for p in self.paths], dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.resolver.resolve(self.paths[index])
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        labels = self.targets[index].copy()
        valid = ((labels == 0) | (labels == 1)).astype(np.float32)
        labels = np.where(valid > 0, labels, 0).astype(np.float32)
        return {
            "image": image,
            "target": torch.from_numpy(labels),
            "valid": torch.from_numpy(valid),
            "domain": int(self.domains[index]),
            "path": self.paths[index],
            "index": int(index),
        }


def make_domain_balanced_sampler(domains: np.ndarray, seed: int = 42):
    domains = np.asarray(domains, dtype=np.int64)
    known = domains >= 0
    counts = {d: int((domains == d).sum()) for d in sorted(set(domains[known].tolist()))}
    weights = np.ones(len(domains), dtype=np.float64)
    for d, count in counts.items():
        if count > 0:
            weights[domains == d] = 1.0 / count
    if (~known).any():
        weights[~known] = 1.0 / max(int((~known).sum()), 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(domains),
        replacement=True,
        generator=generator,
    )


class MixStyle(nn.Module):
    """Feature-statistics mixing for source-domain generalization."""
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x):
        if not self.training or self.p <= 0 or torch.rand(1, device=x.device).item() > self.p:
            return x
        if x.ndim != 4 or x.size(0) < 2:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu_det, sig_det = mu.detach(), sig.detach()
        x_norm = (x - mu_det) / sig_det
        perm = torch.randperm(x.size(0), device=x.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((x.size(0), 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        mu_mix = lam * mu_det + (1.0 - lam) * mu_det[perm]
        sig_mix = lam * sig_det + (1.0 - lam) * sig_det[perm]
        return x_norm * sig_mix + mu_mix


class EfficientNetB0DG(nn.Module):
    def __init__(self, num_attributes: int = NUM_ATTRIBUTES, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            base = efficientnet_b0(weights=weights)
        except Exception:
            base = efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=0.25)
        in_features = base.classifier[1].in_features
        self.attr_head = nn.Linear(in_features, num_attributes)
        self.domain_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )
        self.mixstyle1 = MixStyle(p=0.45, alpha=0.1)
        self.mixstyle2 = MixStyle(p=0.35, alpha=0.1)

    def extract_features(self, x):
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx == 2:
                x = self.mixstyle1(x)
            elif idx == 4:
                x = self.mixstyle2(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x, grl_lambda: float = 0.0, return_domain: bool = False):
        feat = self.extract_features(x)
        attr_logits = self.attr_head(self.dropout(feat))
        if not return_domain:
            return attr_logits
        rev = grad_reverse(feat, grl_lambda)
        domain_logits = self.domain_head(rev)
        return attr_logits, domain_logits, feat


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradientReverse.apply(x, lambd)


class MaskedAsymmetricLoss(nn.Module):
    """
    ASL-style multi-label loss. Unlike large BCE pos_weight, this does not
    aggressively push rare labels positive, which helps precision under shift.
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits, targets, valid):
        targets = targets.float()
        valid = valid.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss = loss + (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)
        loss = -loss * valid
        return loss.sum() / valid.sum().clamp(min=1.0)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_v in ema_state.items():
            model_v = model_state[key].detach()
            if not torch.is_floating_point(ema_v):
                ema_v.copy_(model_v)
            else:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


@dataclass
class TrainConfig:
    image_height: int = 256
    image_width: int = 128
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 8
    lr: float = 2e-4
    weight_decay: float = 2e-4
    domain_loss_weight: float = 0.08
    grl_lambda: float = 0.35
    ema_decay: float = 0.9995
    num_workers: int = 2


def domain_loss(domain_logits, domains):
    domains = domains.long()
    valid = domains >= 0
    if valid.sum() == 0:
        return domain_logits.sum() * 0.0
    return F.cross_entropy(domain_logits[valid], domains[valid])
