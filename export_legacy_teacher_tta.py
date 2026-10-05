from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--official-root", type=Path, default=Path("/content/UPAR-Challenge-2027")
    )
    p.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints/distill_teacher_convnext_small_seed42_best.pt"),
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Results"),
    )
    p.add_argument("--tta", choices=["flip", "flip-scale"], default="flip")
    p.add_argument("--calibration-bias", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-name", default="Submission_LEGACY_TEACHER_TTA.zip")
    return p.parse_args()


def half_state(state):
    out = {}
    for k, v in state.items():
        if not k.startswith(("features.", "avgpool.", "norm.", "attr_head.")):
            continue
        t = v.detach().cpu()
        if torch.is_floating_point(t):
            t = t.half()
        out[k] = t
    return out


def main():
    args = parse_args()
    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(args.teacher_checkpoint)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("inference_model_state") or ck.get("ema_model_state")
    thresholds = ck.get("thresholds")
    attrs = ck.get("attribute_names")
    if state is None or thresholds is None or attrs is None:
        raise RuntimeError(
            "Teacher checkpoint must contain state, calibrated thresholds, and attribute_names. "
            "Resume the old teacher cell once so final calibration is written."
        )

    out = Path("/content/submission_legacy_teacher_tta")
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)

    pkg = {
        "format": "upar-legacy-teacher-tta-v1",
        "state_dict": half_state(state),
        "thresholds": torch.as_tensor(thresholds, dtype=torch.float32),
        "attribute_names": attrs,
        "image_height": int(ck.get("image_height", 288)),
        "image_width": int(ck.get("image_width", 144)),
        "tta": args.tta,
        "calibration_bias": float(args.calibration_bias),
    }
    torch.save(pkg, out / "assets" / "model.pt")
    (out / "assets" / "config.json").write_text(
        json.dumps(
            {
                "architecture": "convnext_small",
                "storage_dtype": "float16",
                "tta": args.tta,
                "calibration_bias": args.calibration_bias,
                "batch_size": args.batch_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = args.official_root / "examples/task1/sample_code_submission/metadata.yaml"
    if not meta.exists():
        raise FileNotFoundError(meta)
    shutil.copy2(meta, out / "metadata.yaml")
    (out / "NOTICE.md").write_text(
        "UPAR Track-1: direct deployment of the stronger ConvNeXt-Small teacher with TTA.\n",
        encoding="utf-8",
    )
    (out / "LICENSE-model.txt").write_text(
        "Participant-trained weights; torchvision components retain upstream licenses.\n",
        encoding="utf-8",
    )

    run_py = r'''from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small

ROOT=Path(__file__).resolve().parent
PKG=torch.load(ROOT/"assets"/"model.pt",map_location="cpu",weights_only=False)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
H=int(PKG["image_height"]); W=int(PKG["image_width"])
TTA=str(PKG.get("tta","flip"))
BIAS=float(PKG.get("calibration_bias",0.0))
BATCH_SIZE=int(os.environ.get("UPAR_BATCH_SIZE","32"))
ATTRS=PKG["attribute_names"]

TFM=transforms.Compose([
    transforms.Resize((H,W),interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        b=convnext_small(weights=None)
        self.features=b.features
        self.avgpool=b.avgpool
        self.norm=b.classifier[0]
        self.dropout=nn.Dropout(0.30)
        self.attr_head=nn.Linear(b.classifier[2].in_features,40)
    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x=self.norm(x)
        x=torch.flatten(x,1)
        return self.attr_head(self.dropout(x))

MODEL=Net()
MODEL.load_state_dict(PKG["state_dict"],strict=True)
MODEL=MODEL.to(DEVICE).eval()

th=torch.as_tensor(PKG["thresholds"],dtype=torch.float32,device=DEVICE).clamp(1e-5,1-1e-5)
THLOGIT=torch.logit(th)

def _path(s):
    for k in ("image_path","path","image","# image","filename"):
        if k in s and s[k] is not None: return str(s[k])
    raise KeyError(list(s.keys()))

def _requested(s):
    r=s.get("attribute_names") or s.get("attributes")
    return r if r else ATTRS

def _load(s):
    with Image.open(_path(s)) as im:
        return TFM(im.convert("RGB"))

def _scale(x):
    hh,ww=H+24,W+12
    y=F.interpolate(x,size=(hh,ww),mode="bicubic",align_corners=False)
    t=(hh-H)//2; l=(ww-W)//2
    return y[:,:,t:t+H,l:l+W]

@torch.inference_mode()
def _predict(x):
    views=[x,torch.flip(x,dims=[3])]
    if TTA=="flip-scale":
        xs=_scale(x)
        views += [xs,torch.flip(xs,dims=[3])]
    z=None
    for v in views:
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
            q=MODEL(v).float()
        z=q if z is None else z+q
    z=z/len(views)
    z=z-THLOGIT.reshape(1,-1)-BIAS
    return torch.sigmoid(z)

def _reorder(row,req):
    if list(req)==list(ATTRS): return row
    lut={a:i for i,a in enumerate(ATTRS)}
    return [row[lut[a]] for a in req]

def predict_batch(samples):
    out=[]
    for st in range(0,len(samples),BATCH_SIZE):
        chunk=samples[st:st+BATCH_SIZE]
        x=torch.stack([_load(s) for s in chunk]).to(DEVICE,non_blocking=True)
        p=_predict(x).cpu().numpy()
        for s,row in zip(chunk,p):
            out.append(_reorder(row.tolist(),_requested(s)))
    return out

def predict_image(sample): return predict_batch([sample])[0]
def load_model(): return None
'''
    (out / "run.py").write_text(run_py, encoding="utf-8")

    final = args.result_dir / args.output_name
    if final.exists():
        final.unlink()
    with zipfile.ZipFile(final, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc",".pyo"}:
                z.write(p, p.relative_to(out))
    print("Built:", final, "MiB:", final.stat().st_size/1024/1024)


if __name__ == "__main__":
    main()
