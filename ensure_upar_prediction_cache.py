"""Incrementally build missing/stale source validation inference cache for a frozen ensemble.

Never recomputes baseline calibration or trains model weights. Caches are saved atomically.
"""
from __future__ import annotations
import argparse, os, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from strong_par import UPARDataset, ImageResolver, StrongPARModel, build_eval_transform
from prototype_par import HybridPrototypePAR
from dinov2_par import DinoAttributePAR
from calibrate_generalization import get_state, load_pt
from calibrate_shift_robust import signature, infer_logits


def run(args):
    cal=load_pt(args.base_calibration)
    if cal.get('tta')!='flip':raise ValueError('Expected flip TTA source calibration')
    entries=[e for e in cal['entries'] if float(cal['weights'].get(e['key'],0.))>1e-7]
    if not entries:raise ValueError('No active model entries')
    df=pd.read_csv(args.repo_root/'data/annotations/task1/val/gt.csv')
    print('[DATA] Resolving source validation images',flush=True)
    ds=UPARDataset(df,ImageResolver(args.repo_root/'data',args.repo_root),build_eval_transform(288,144))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=device.type=='cuda')
    args.result_dir.mkdir(parents=True,exist_ok=True)
    for e in entries:
        path=Path(e['checkpoint'])
        if not path.is_file():raise FileNotFoundError(f'Missing checkpoint: {path}')
        dst=args.result_dir/f"shift_predictions_{e['kind']}_seed{e['seed']}_flip.npz"
        sig=signature(path,'flip',args.repo_root)
        current=False
        if dst.is_file() and not args.force:
            try:
                with np.load(dst,allow_pickle=False) as z:
                    current=(str(z['signature'].item())==sig and
                             len(z['y'])==len(ds) and
                             np.array_equal(z['domain'],ds.domains))
            except Exception:pass
        if current:
            print('[CACHE HIT]',dst.name,flush=True)
            continue
        ck=load_pt(path)
        if e['kind']=='strong':model=StrongPARModel(backbone='convnext_tiny',pretrained=False)
        elif e['kind']=='dinov2':model=DinoAttributePAR(pretrained=False,**dict(e.get('config') or {}))
        else:model=HybridPrototypePAR(pretrained=False,**dict(e.get('config') or {}))
        model.load_state_dict(get_state(ck),strict=True)
        del ck
        model.to(device).eval()
        print('[INFERENCE]',e['key'],f'{len(ds)} images',flush=True)
        record=infer_logits(model,loader,device,'flip',e['key'])
        if not np.array_equal(record['domain'],ds.domains):
            raise RuntimeError('Inference output domain mismatch')
        fd,tmp=tempfile.mkstemp(prefix='._pred_',suffix='.npz',dir=str(args.result_dir))
        os.close(fd)
        try:
            np.savez_compressed(tmp,signature=sig,**record)
            os.replace(tmp,dst)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        print('[CACHE SAVED]',dst,flush=True)
        del model,record
        if device.type=='cuda':torch.cuda.empty_cache()
    print('[READY] All selected-model FIT/AUDIT prediction caches',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-calibration',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
