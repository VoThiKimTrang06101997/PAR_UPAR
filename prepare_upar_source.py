"""Validate / prepare official UPAR source images without deleting user data.

Uses official UPAR-Challenge-2027 download_datasets.py only; not hidden labels.
"""
from __future__ import annotations
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from strong_par import detect_image_column, infer_domain_id, ImageResolver


def contains_images(folder:Path):
    return folder.is_dir() and any(p.is_file() for p in folder.rglob('*.jpg'))


def run(root:Path, auto_prepare:bool):
    if not root.is_dir():
        if not auto_prepare:
            raise FileNotFoundError(root)
        print('[PREPARE] Cloning official UPAR source metadata',flush=True)
        import subprocess
        subprocess.run(['git','clone','--depth','1',
                        'https://github.com/speckean/UPAR-Challenge-2027.git',str(root)],check=True)
    data=root/'data'
    ann=data/'annotations/task1/val/gt.csv'
    if not ann.is_file():raise FileNotFoundError(f'Official validation annotations missing: {ann}')
    df=pd.read_csv(ann)
    paths=df[detect_image_column(df)].astype(str).tolist()
    sample={}
    for value in paths:
        d=infer_domain_id(value)
        if d in (0,1,2) and d not in sample:sample[d]=value
    if set(sample)!={0,1,2}:raise RuntimeError('Expected all 3 source domains in validation annotations')
    resolver=ImageResolver(data,root)
    missing=[]
    for d,value in sample.items():
        try:
            path=resolver.resolve(value)
            print(f'[FOUND] source domain{d}: {path}',flush=True)
        except FileNotFoundError:
            missing.append(d)
            print(f'[MISSING] source domain{d}: {value}',flush=True)
    if missing and auto_prepare:
        loader=root/'download_datasets.py'
        if not loader.is_file():raise FileNotFoundError(f'Official download_datasets.py missing: {loader}')
        sys.path.insert(0,str(root))
        import download_datasets as dd
        fn={0:('Market1501',dd.prepare_market),1:('PA100k',dd.prepare_pa100k),2:('PETA',dd.prepare_peta)}
        for d in missing:
            name,prepare=fn[d]
            folder=data/name
            if folder.exists() and not contains_images(folder):
                backup=folder.with_name(folder.name+'_empty_backup')
                if backup.exists():
                    raise RuntimeError(f'Empty directory backup already exists: {backup}; inspect before retry')
                folder.rename(backup)
                print('[BACKUP] Empty/missing-image directory ->',backup,flush=True)
            print(f'[DOWNLOAD] Using official source downloader for {name}',flush=True)
            prepare(data)
    # Resolve all validation samples before GPU training/inference, to avoid 20-minute failures.
    resolver=ImageResolver(data,root)
    for value in tqdm(paths,desc='Verify all annotated validation images',unit='image',dynamic_ncols=True):
        resolver.resolve(value)
    print(f'[DATA READY] Verified {len(paths)} validation source images. No private labels used.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--auto-prepare',action='store_true')
    a=parser.parse_args()
    run(a.repo_root,a.auto_prepare)
