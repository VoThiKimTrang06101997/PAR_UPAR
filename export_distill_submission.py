from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import torch


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--official-root",type=Path,default=Path("/content/UPAR-Challenge-2027"))
    p.add_argument("--checkpoint-dir",type=Path,default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints"))
    p.add_argument("--result-dir",type=Path,default=Path("/content/drive/MyDrive/PedestrianAttributeRecognition/Results"))
    p.add_argument("--seeds",default="42,123")
    p.add_argument("--tta",action="store_true")
    p.add_argument("--batch-size",type=int,default=48)
    return p.parse_args()


def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def main():
    args=parse_args(); args.result_dir.mkdir(parents=True,exist_ok=True)
    out=Path("/content/submission_distill_convnext")
    if out.exists(): shutil.rmtree(out)
    (out/"assets").mkdir(parents=True)
    members=[]
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        p=args.checkpoint_dir/f"distill_student_convnext_tiny_seed{seed}_best.pt"
        if not p.exists():
            print("Skipping missing checkpoint:",p); continue
        ck=torch.load(p,map_location="cpu",weights_only=False)
        st=ck.get("inference_model_state") or ck.get("ema_model_state")
        # Keep only inference network keys; domain head and MixStyle have no runtime value.
        st={k:v.cpu() for k,v in st.items() if not k.startswith("domain_head.") and not k.startswith("mix")}
        th=torch.as_tensor(ck.get("thresholds",torch.full((40,),0.5)),dtype=torch.float32).reshape(-1)
        members.append({"seed":seed,"state_dict":st,"thresholds":th,"metrics":ck.get("metrics_calibrated",{})})
    if not members: raise RuntimeError("No distill student checkpoints found")
    package={
        "format":"upar-distill-convnext-inference-v1",
        "architecture":"convnext_tiny",
        "members":members,
        "image_height":288,"image_width":144,
        "tta_horizontal_flip":bool(args.tta),
        "attribute_names":members and torch.load(args.checkpoint_dir/f"distill_student_convnext_tiny_seed{members[0]['seed']}_best.pt",map_location="cpu",weights_only=False).get("attribute_names"),
    }
    torch.save(package,out/"assets"/"model.pt")
    (out/"assets"/"config.json").write_text(json.dumps({
        "architecture":"convnext_tiny","ensemble":len(members),"image_height":288,"image_width":144,
        "tta_horizontal_flip":bool(args.tta),"batch_size":args.batch_size,
    },indent=2),encoding="utf-8")

    sample_meta=args.official_root/"examples"/"task1"/"sample_code_submission"/"metadata.yaml"
    if not sample_meta.exists(): raise FileNotFoundError(sample_meta)
    shutil.copy2(sample_meta,out/"metadata.yaml")
    (out/"NOTICE.md").write_text("UPAR 2027 ConvNeXt-Tiny student ensemble distilled from ConvNeXt-Small teacher. Offline inference only.\n",encoding="utf-8")
    (out/"LICENSE-model.txt").write_text("Participant-trained weights. Torchvision backbone components retain their upstream licenses.\n",encoding="utf-8")

    run_py=r'''from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny

ROOT=Path(__file__).resolve().parent
PKG=torch.load(ROOT/"assets"/"model.pt",map_location="cpu",weights_only=False)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
H=int(PKG.get("image_height",288)); W=int(PKG.get("image_width",144))
BATCH_SIZE=int(os.environ.get("UPAR_BATCH_SIZE","48"))
ATTRS=PKG["attribute_names"]
TTA=bool(PKG.get("tta_horizontal_flip",False))

TFM=transforms.Compose([
    transforms.Resize((H,W),interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        b=convnext_tiny(weights=None)
        self.features=b.features; self.avgpool=b.avgpool; self.norm=b.classifier[0]
        self.dropout=nn.Identity(); self.attr_head=nn.Linear(b.classifier[2].in_features,40)
    def forward(self,x):
        x=self.features(x); x=self.avgpool(x); x=self.norm(x); x=torch.flatten(x,1)
        return self.attr_head(x)

MODELS=[]; TH=[]
for item in PKG["members"]:
    m=Net()
    # Training model has dropout module with no state, same inference keys otherwise.
    missing,unexpected=m.load_state_dict(item["state_dict"],strict=False)
    bad=[x for x in missing if not x.startswith("dropout")]
    if bad or unexpected:
        raise RuntimeError(f"State mismatch missing={bad} unexpected={unexpected}")
    m=m.to(DEVICE).eval()
    if DEVICE.type=="cuda": m=m.to(memory_format=torch.channels_last)
    MODELS.append(m)
    TH.append(torch.as_tensor(item["thresholds"],dtype=torch.float32,device=DEVICE))

def _path(sample):
    for k in ("image_path","path","image","# image","filename"):
        if k in sample and sample[k] is not None: return str(sample[k])
    raise KeyError(f"No image path in sample keys={list(sample.keys())}")

def _requested(sample):
    req=sample.get("attribute_names") or sample.get("attributes")
    return req if req else ATTRS

def _load(sample):
    with Image.open(_path(sample)) as im: return TFM(im.convert("RGB"))

def _shifted_prob(logits,th):
    # Encode calibrated decision thresholds into probabilities while preserving ranking.
    eps=1e-5
    t=th.clamp(eps,1-eps)
    shift=torch.log(t/(1-t))
    return torch.sigmoid(logits-shift)

@torch.inference_mode()
def _predict(x):
    if DEVICE.type=="cuda": x=x.to(memory_format=torch.channels_last)
    acc=None
    for m,th in zip(MODELS,TH):
        if TTA:
            xx=torch.cat([x,torch.flip(x,dims=[3])],dim=0)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                z=m(xx)
            n=x.size(0); p=0.5*(_shifted_prob(z[:n],th)+_shifted_prob(z[n:],th))
        else:
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                z=m(x)
            p=_shifted_prob(z,th)
        acc=p if acc is None else acc+p
    return acc/len(MODELS)

def _reorder(row,requested):
    if list(requested)==list(ATTRS): return row
    lut={a:i for i,a in enumerate(ATTRS)}
    return [row[lut[a]] for a in requested]

def predict_batch(samples:list[dict])->list[list[float]]:
    out=[]
    for st in range(0,len(samples),BATCH_SIZE):
        chunk=samples[st:st+BATCH_SIZE]
        x=torch.stack([_load(s) for s in chunk]).to(DEVICE,non_blocking=True)
        p=_predict(x).float().cpu().numpy()
        for s,row in zip(chunk,p): out.append(_reorder(row.tolist(),_requested(s)))
    return out

def predict_image(sample:dict)->list[float]:
    return predict_batch([sample])[0]

def load_model():
    return None
'''
    (out/"run.py").write_text(run_py,encoding="utf-8")

    final=args.result_dir/"Submission_DISTILL_CONVNEXT.zip"
    if final.exists(): final.unlink()
    with zipfile.ZipFile(final,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc",".pyo"}:
                z.write(p,p.relative_to(out))
    print("Built:",final)
    print("MiB:",final.stat().st_size/1024/1024)
    print("SHA256:",sha256(final))
    print("Members:",[m["seed"] for m in members],"TTA:",args.tta)

if __name__=="__main__": main()
