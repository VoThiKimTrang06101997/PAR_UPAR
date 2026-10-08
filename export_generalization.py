"""Package the selected audit-checked ensemble as a standalone Codabench ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
import torch


def load_pt(path):
    try:return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(path,map_location='cpu')


def get_state(ck):
    for k in ('inference_model_state','ema_model_state','model_state'):
        if k in ck and ck[k] is not None:return ck[k]
    raise RuntimeError('No deployable checkpoint state')


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--runtime-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--micro-batch-size',type=int,default=8)
    p.add_argument('--require-trained',action='store_true',help='Reject submission if audit guard selected no retrained member')
    args=p.parse_args()
    if not 1<=args.micro_batch_size<=64:raise ValueError('micro batch must be between 1 and 64')
    cal=load_pt(args.calibration)
    if len(cal['attribute_names'])!=40:raise ValueError('Expected 40 attributes')
    if args.require_trained and not cal.get('contains_retrained',False):
        raise RuntimeError('Retrained candidate was rejected by the source-domain audit guard; no new submission exported. Review generalization_audit_report.json.')
    members=[]
    for entry in cal['entries']:
        w=float(cal['weights'].get(entry['key'],0))
        if w<=1e-8:continue
        ck=load_pt(Path(entry['checkpoint']))
        sd=get_state(ck)
        cfg=dict(ck.get('prototype_config',{})) if entry['kind'] in ('prototype','trained') else {}
        cfg.pop('pretrained',None)
        members.append(dict(kind=entry['kind'],seed=int(entry['seed']),weight=w,
                            config=cfg,state_dict={k:(v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu())
                                                       for k,v in sd.items()}))
        del ck,sd
    if not members:raise ValueError('No weighted ensemble members')
    with tempfile.TemporaryDirectory() as root:
        folder=Path(root)/'submission';(folder/'assets').mkdir(parents=True)
        torch.save(dict(members=members,attribute_names=cal['attribute_names'],
                        threshold_logits=cal['threshold_logits'],strength=cal['strength'],
                        image_height=cal['image_height'],image_width=cal['image_width'],tta='flip'),
                   folder/'assets'/'model.pt')
        runtime=args.runtime_file.read_text(encoding='utf-8')
        runtime=runtime.replace("'UPAR_MICRO_BATCH_SIZE', '8'",f"'UPAR_MICRO_BATCH_SIZE', '{int(args.micro_batch_size)}'")
        (folder/'run.py').write_text(runtime,encoding='utf-8')
        (folder/'assets'/'config.json').write_text(json.dumps({
            'name':'UPAR conservative cross-source ensemble',
            'model_types':[f"{m['kind']}:{m['seed']}" for m in members],
            'weights':[float(m['weight']) for m in members],
            'strength':cal['strength'],
            'attribute_names':cal['attribute_names'],
            'audit_metrics':cal.get('audit_metrics'),
            'micro_batch_size':int(args.micro_batch_size),
            'tta':'flip',
        },indent=2),encoding='utf-8')
        (folder/'metadata.yaml').write_text('name: UPAR Generalization Ensemble\nframework: pytorch\n',encoding='utf-8')
        (folder/'NOTICE.md').write_text('Prototype and optional ConvNeXt-Tiny models. Flip TTA. Fit/audit split calibration.\n',encoding='utf-8')
        (folder/'LICENSE-model.txt').write_text('Participant-trained models; pretrained components retain upstream licenses.\n',encoding='utf-8')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():args.output.unlink()
        with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(folder.rglob('*')):
                if f.is_file():z.write(f,str(f.relative_to(folder)))
    expected={'LICENSE-model.txt','NOTICE.md','assets/config.json','assets/model.pt','metadata.yaml','run.py'}
    with zipfile.ZipFile(args.output) as z:
        assert set(z.namelist())==expected, z.namelist()
    print('Output:',args.output,'MiB:',round(args.output.stat().st_size/1024**2,2))
    print('SHA256:',sha256(args.output))
    print('Members:',[(x['kind'],x['seed'],x['weight']) for x in members],flush=True)
    smoke_test(args.output)
    print('SUBMISSION_RUNTIME_CHECK: PASS',flush=True)


def smoke_test(zippath):
    """Exercise the *exact exported archive*, including load_model and both APIs."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(zippath) as z:z.extractall(root)
        subprocess.run([sys.executable,'-m','py_compile',str(root/'run.py')],check=True)
        image=root/'probe.jpg';Image.new('RGB',(144,288),(128,128,128)).save(image)
        cfg=json.loads((root/'assets/config.json').read_text())
        src=f'''import sys,math\nsys.path.insert(0,{str(root)!r})\nimport run\nsample={{'image_path':{str(image)!r},'attribute_names':{cfg['attribute_names']!r}}}\nrun.load_model()\na=run.predict_image(sample)\nb=run.predict_batch([sample,sample])\nassert len(a)==40 and len(b)==2 and all(len(r)==40 for r in b)\nassert all(math.isfinite(float(x)) and 0<=float(x)<=1 for x in a)\nprint('RUNTIME_SMOKE_TEST_OK')\n'''
        proc=subprocess.run([sys.executable,'-c',src],cwd=root,text=True,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'UPAR_MICRO_BATCH_SIZE':'1'})
        if proc.returncode:
            raise RuntimeError('Exported ZIP failed runtime test:\n'+proc.stdout+'\n'+proc.stderr[-5000:])
        print(proc.stdout.strip(),flush=True)


if __name__=='__main__':main()
