from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image, ImageFile
from torchvision import transforms

from .config import ATTRIBUTE_NAMES, DOMAIN_TO_ID

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_upar_csv(csv_path):
    df = pd.read_csv(csv_path)
    image_col = df.columns[0]
    missing = [c for c in ATTRIBUTE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing UPAR attributes in {csv_path}: {missing}")
    return df, image_col


def infer_domain_id(rel_path: str) -> int:
    prefix = str(rel_path).replace("\\", "/").split("/")[0]
    if prefix not in DOMAIN_TO_ID:
        return -1
    return DOMAIN_TO_ID[prefix]


class UPARAttributeDataset(Dataset):
    def __init__(self, csv_path, data_root, max_samples=0):
        self.csv_path = str(csv_path)
        self.data_root = Path(data_root)
        self.df, self.image_col = read_upar_csv(csv_path)
        if max_samples and max_samples > 0:
            self.df = self.df.iloc[:max_samples].copy()
        self.paths = self.df[self.image_col].astype(str).tolist()
        self.targets = self.df[ATTRIBUTE_NAMES].to_numpy(np.float32)
        self.domains = np.array([infer_domain_id(p) for p in self.paths], dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def _resolve(self, rel):
        p = self.data_root / rel
        if p.exists():
            return p
        alt = self.data_root.parent / rel
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Image not found: {rel}\nTried: {p}\nTried: {alt}")

    def __getitem__(self, idx):
        rel = self.paths[idx]
        img = Image.open(self._resolve(rel)).convert("RGB")
        return {
            "image": img,
            "target": torch.from_numpy(self.targets[idx]),
            "domain": int(self.domains[idx]),
            "path": rel,
        }


class TemplateInferenceDataset(Dataset):
    def __init__(self, template_csv, image_root):
        self.template_csv = str(template_csv)
        self.image_root = Path(image_root)
        self.df = pd.read_csv(template_csv)
        self.image_col = self.df.columns[0]
        self.paths = self.df[self.image_col].astype(str).tolist()
        self._basename_index = None

    def __len__(self):
        return len(self.paths)

    def _build_index(self):
        self._basename_index = {}
        for p in self.image_root.rglob("*"):
            if p.is_file():
                self._basename_index.setdefault(p.name, p)

    def _resolve(self, rel):
        direct = self.image_root / rel
        if direct.exists():
            return direct
        parts = Path(rel).parts
        if len(parts) > 1:
            alt = self.image_root.joinpath(*parts[1:])
            if alt.exists():
                return alt
        if self._basename_index is None:
            self._build_index()
        p = self._basename_index.get(Path(rel).name)
        if p is None:
            raise FileNotFoundError(f"Could not resolve test image: {rel} under {self.image_root}")
        return p

    def __getitem__(self, idx):
        rel = self.paths[idx]
        img = Image.open(self._resolve(rel)).convert("RGB")
        return {"image": img, "path": rel}


def make_augmentations():
    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    strong = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.30, contrast=0.30, saturation=0.25, hue=0.05
        )], p=0.8),
        transforms.RandomGrayscale(p=0.10),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.20),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.15),
    ])
    return weak, strong


class TrainCollator:
    def __init__(self, processor):
        self.processor = processor
        self.weak_aug, self.strong_aug = make_augmentations()

    def __call__(self, batch):
        weak_imgs = [self.weak_aug(x["image"]) for x in batch]
        strong_imgs = [self.strong_aug(x["image"]) for x in batch]
        weak_inputs = self.processor(images=weak_imgs, return_tensors="pt")
        strong_inputs = self.processor(images=strong_imgs, return_tensors="pt")
        return {
            "weak": dict(weak_inputs),
            "strong": dict(strong_inputs),
            "targets": torch.stack([x["target"] for x in batch]),
            "domains": torch.tensor([x["domain"] for x in batch], dtype=torch.long),
            "paths": [x["path"] for x in batch],
        }


class EvalCollator:
    def __init__(self, processor, has_targets=True):
        self.processor = processor
        self.has_targets = has_targets

    def __call__(self, batch):
        inputs = self.processor(images=[x["image"] for x in batch], return_tensors="pt")
        out = {"inputs": dict(inputs), "paths": [x["path"] for x in batch]}
        if self.has_targets:
            out["targets"] = torch.stack([x["target"] for x in batch])
            out["domains"] = torch.tensor([x["domain"] for x in batch], dtype=torch.long)
        return out


class ResumableEpochBatchSampler(Sampler):
    """Deterministic epoch shuffle that can START directly at a saved batch.

    This is the critical resume fix. The old implementation enumerated the DataLoader
    from batch 0 and merely `continue`d until the saved batch. That did not redo
    optimizer steps, but it still decoded/preprocessed thousands of already-consumed
    batches and made tqdm look like training restarted from 0.

    Here the sampler itself removes already-consumed batches BEFORE DataLoader workers
    are started, so resume at batch 4000 begins fetching batch 4000 immediately.
    """

    def __init__(self, data_source, batch_size, seed=605, drop_last=False):
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_batch = 0

    @property
    def total_batches(self):
        n = len(self.data_source)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def set_epoch(self, epoch, start_batch=0):
        self.epoch = int(epoch)
        self.start_batch = max(0, int(start_batch))
        if self.start_batch > self.total_batches:
            raise ValueError(
                f"start_batch={self.start_batch} exceeds total_batches={self.total_batches}"
            )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.data_source), generator=g).tolist()

        first = self.start_batch * self.batch_size
        for begin in range(first, len(order), self.batch_size):
            batch = order[begin: begin + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self):
        return max(0, self.total_batches - self.start_batch)


# Kept for backwards imports in older notebooks, but new training uses the batch sampler above.
class EpochRandomSampler(Sampler):
    def __init__(self, data_source, seed=605):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=g).tolist()

    def __len__(self):
        return len(self.data_source)
