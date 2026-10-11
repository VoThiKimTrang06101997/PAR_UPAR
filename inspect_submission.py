"""Compare exported UPAR ZIPs semantically; never confuse an anchor with a new candidate."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import torch


def read_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        expected={'run.py','metadata.yaml','assets/model.pt','assets/config.json','NOTICE.md','LICENSE-model.txt'}
        if names!=expected: raise ValueError(f'Archive contents mismatch: {sorted(names^expected)}')
        if z.testzip() is not None: raise ValueError(f'Corrupt archive: {path}')
        import io
        obj=torch.load(io.BytesIO(z.read('assets/model.pt')),map_location='cpu',weights_only=False)
        cfg=json.loads(z.read('assets/config.json'))
    digest=hashlib.sha256()
    digest.update(str(obj.get('tta','flip')).encode())
    digest.update(str(obj.get('strength',1.)).encode())
    for tkey in ('threshold_logits',):
        t=obj[tkey]
        digest.update(torch.as_tensor(t).float().contiguous().numpy().tobytes())
    details=[]
    for m in obj['members']:
        details.append({'model':f"{m['kind']}:{m['seed']}", 'weight':float(m['weight'])})
        digest.update(f"{m['kind']}:{m['seed']}:{m['weight']:.10f}".encode())
        for k,t in sorted(m['state_dict'].items()):
            digest.update(k.encode())
            digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    return {'path':str(path),'semantic_sha256':digest.hexdigest(),
            'members':details,'tta':obj.get('tta','flip'),
            'threshold_range':[float(torch.sigmoid(torch.as_tensor(obj['threshold_logits'])).min()),
                               float(torch.sigmoid(torch.as_tensor(obj['threshold_logits'])).max())],
            'size_mb':round(path.stat().st_size/1024**2,2),'config':cfg}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--experimental',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    a=read_zip(args.baseline);b=read_zip(args.experimental)
    unique=a['semantic_sha256']!=b['semantic_sha256']
    report={'distinct_inference_configuration':unique,'baseline':a,'experimental':b}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
    print('BASELINE:',a['members'],'TTA',a['tta'])
    print('EXPERIMENT:',b['members'],'TTA',b['tta'])
    print('DISTINCT_INFERENCE_CONFIGURATION:',unique,flush=True)
    if not unique:
        raise RuntimeError('Candidate duplicates baseline; do NOT waste a Codabench submission.')

if __name__=='__main__':main()
