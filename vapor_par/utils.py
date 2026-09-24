import csv, os, random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed=605):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_inputs(inputs, device):
    return {k:v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in inputs.items()}

def append_csv(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())

def get_amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
