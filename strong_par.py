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
    swin_v2_t, Swin_V2_T_Weights,
)
from tqdm.auto import tqdm


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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
        scan = tqdm(
            self.data_root.rglob("*"),
            desc="Indexing image files",
            unit="file",
            dynamic_ncols=True,
            leave=True,
        )
        for p in scan:
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                self._index.setdefault(p.name, p)
        scan.close()
        print(f"Image index ready: {len(self._index):,} files", flush=True)

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
    """
    Full-body preserving augmentation.

    The previous RandomResizedCrop(scale=0.74) could remove head/feet, which is
    especially damaging for age, hair, hat and lower-body attributes.
    """
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=4.0,
                translate=(0.025, 0.025),
                scale=(0.96, 1.04),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
        ], p=0.35),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.20, contrast=0.20, saturation=0.16, hue=0.025
            )
        ], p=0.65),
        transforms.RandomAutocontrast(p=0.08),
        transforms.RandomGrayscale(p=0.025),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        transforms.RandomErasing(
            p=0.14, scale=(0.01,0.075), ratio=(0.40,2.5), value="random"
        ),
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
        raw_paths = self.df[self.image_col].astype(str).tolist()
        self.targets = self.df[ATTRIBUTE_NAMES].to_numpy(dtype=np.float32)
        self.domains = np.asarray([infer_domain_id(p) for p in raw_paths], dtype=np.int64)
        self.transform = transform

        self.paths = []
        for value in tqdm(
            raw_paths,
            desc="Resolving image paths",
            unit="img",
            dynamic_ncols=True,
            leave=True,
        ):
            self.paths.append(str(resolver.resolve(value)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
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


def make_tempered_domain_sampler(domains: np.ndarray, seed=42, alpha=0.35):
    """
    Partial domain rebalance.

    alpha=0   -> original sample frequency
    alpha=1   -> exactly equal domain mass (the old patch)
    alpha=0.35 keeps PA100K important while giving Market/PETA more exposure.
    """
    domains = np.asarray(domains, dtype=np.int64)
    weights = np.ones(len(domains), dtype=np.float64)
    for d in sorted(set(int(x) for x in domains.tolist() if int(x) >= 0)):
        n = int((domains == d).sum())
        if n:
            weights[domains == d] = float(n) ** (-float(alpha))
    gen = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=gen,
    )


def dataset_pos_weight(targets: np.ndarray, exponent=0.25, max_weight=2.5):
    y = np.asarray(targets, dtype=np.float32)
    out = np.ones(y.shape[1], dtype=np.float32)
    for c in range(y.shape[1]):
        valid = (y[:, c] == 0) | (y[:, c] == 1)
        yc = y[valid, c]
        pos = float((yc == 1).sum())
        neg = float((yc == 0).sum())
        if pos > 0 and neg > 0:
            ratio = (neg + 1.0) / (pos + 1.0)
            out[c] = np.clip(ratio ** float(exponent), 0.65, float(max_weight))
    return torch.from_numpy(out)


class StrongPARModel(nn.Module):
    def __init__(self, backbone="convnext_tiny", num_attributes=NUM_ATTRIBUTES, pretrained=True):
        super().__init__()
        self.backbone_name = str(backbone)
        if self.backbone_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            try:
                base = convnext_tiny(weights=weights)
            except Exception:
                base = convnext_tiny(weights=None)
            in_features = int(base.classifier[2].in_features)
            base.classifier[2] = nn.Sequential(
                nn.Dropout(0.22),
                nn.Linear(in_features, num_attributes),
            )
            self.net = base
        elif self.backbone_name == "swin_v2_t":
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            try:
                base = swin_v2_t(weights=weights)
            except Exception:
                base = swin_v2_t(weights=None)
            in_features = int(base.head.in_features)
            base.head = nn.Sequential(
                nn.Dropout(0.22),
                nn.Linear(in_features, num_attributes),
            )
            self.net = base
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

    def forward(self, x):
        return self.net(x)


class LegacyConvNeXtSmallTeacher(nn.Module):
    """Matches the ConvNeXt-Small teacher created by the previous notebook."""
    def __init__(self):
        super().__init__()
        base = convnext_small(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = copy.deepcopy(base.classifier[0])
        in_features = int(base.classifier[2].in_features)
        self.dropout = nn.Dropout(0.30)
        self.attr_head = nn.Linear(in_features, NUM_ATTRIBUTES)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        return self.attr_head(self.dropout(x))


def load_legacy_teacher(checkpoint_path: str | Path, device):
    p = Path(checkpoint_path)
    if not p.exists():
        return None
    obj = torch.load(p, map_location="cpu", weights_only=False)
    state = obj.get("inference_model_state") or obj.get("ema_model_state") or obj.get("model_state") or obj
    state = {
        k: v for k, v in state.items()
        if k.startswith(("features.", "avgpool.", "norm.", "attr_head."))
    }
    teacher = LegacyConvNeXtSmallTeacher()
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("dropout")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Legacy teacher state mismatch: missing={bad_missing}, unexpected={unexpected}"
        )
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


class MaskedBalancedBCE(nn.Module):
    def __init__(self, pos_weight, label_smoothing=0.03):
        super().__init__()
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets, valid):
        targets = targets.float()
        valid = valid.float()
        if self.label_smoothing > 0:
            s = self.label_smoothing
            soft_targets = targets * (1.0 - s) + 0.5 * s
        else:
            soft_targets = targets
        raw = F.binary_cross_entropy_with_logits(
            logits,
            soft_targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        raw = raw * valid
        return raw.sum() / valid.sum().clamp(min=1.0)


def response_kd_loss(student_logits, teacher_logits, valid, temperature=1.5):
    """Small response-level KD term; unlike the old patch it cannot dominate BCE."""
    t = float(temperature)
    teacher_p = torch.sigmoid(teacher_logits.detach() / t)
    raw = F.binary_cross_entropy_with_logits(
        student_logits / t, teacher_p, reduction="none"
    ) * (t * t)
    # Only distill labels that exist in the official harmonized training target.
    w = valid.float()
    return (raw * w).sum() / w.sum().clamp(min=1.0)


class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.9995):
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


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No states to average")
    keys = states[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in states]
        if torch.is_floating_point(vals[0]):
            acc = vals[0].detach().float().clone()
            for v in vals[1:]:
                acc.add_(v.detach().float())
            acc.div_(len(vals))
            out[k] = acc.to(vals[0].dtype)
        else:
            out[k] = vals[0].detach().clone()
    return out


@dataclass
class TrainConfig:
    image_height: int = 288
    image_width: int = 144
    batch_size: int = 48
    num_workers: int = 4
    epochs: int = 12
    lr_convnext: float = 1.6e-4
    lr_swin: float = 1.2e-4
    weight_decay: float = 0.035
    ema_decay: float = 0.9995
    sampler_alpha: float = 0.35
    kd_weight: float = 0.12
    kd_temperature: float = 1.5
    kd_warmup_epochs: int = 2
    topk_soup: int = 3

