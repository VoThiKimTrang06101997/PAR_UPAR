"""Conservative, audit-split calibration for UPAR Track 1.

Does not use the private test labels; does not claim true leave-one-domain-out
training (all existing checkpoints were trained on the three source domains).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from strong_par import (ATTRIBUTE_NAMES, ImageResolver, UPARDataset,
                        StrongPARModel, build_eval_transform, seed_everything)
from prototype_par import HybridPrototypePAR
from vapor_par.metrics import evaluate_probs


def load_pt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def get_state(ck):
    for key in ('inference_model_state', 'ema_model_state', 'model_state'):
        if key in ck and ck[key] is not None:
            return ck[key]
    raise ValueError('No model state in checkpoint')


def checkpoint_file(root, kind, seed):
    if kind == 'prototype':
        return root / f'prototype_convnext_tiny_seed{seed}_best.pt'
    if kind == 'trained':
        return root / f'prototype_contrastive_convnext_tiny_seed{seed}_best.pt'
    if kind == 'strong':
        return root / f'strong_convnext_tiny_seed{seed}_best.pt'
    raise ValueError(kind)


def stable_split(paths, domains, audit_fraction=.30):
    """Split by stable filename hash; no label leakage into selection split."""
    split = np.zeros(len(paths), dtype=bool)
    for i, path in enumerate(paths):
        key = str(path).replace('\\', '/').lower().encode('utf-8')
        val = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') / 2**64
        split[i] = val < audit_fraction
    for d in np.unique(domains):
        if d >= 0 and (sum((domains == d) & split) < 20 or sum((domains == d) & ~split) < 20):
            raise RuntimeError(f'Insufficient fit/audit examples for domain {d}')
    return ~split, split


def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -35, 35)))


def logit(x):
    x = np.clip(x, 1e-5, 1-1e-5)
    return np.log(x/(1-x))


def state_signature(path):
    s = path.stat()
    return f'{str(path.resolve())}::{s.st_size}::{s.st_mtime_ns}'


@torch.inference_mode()
def inference(model, kind, loader, device):
    lin, pro, gates, ys, dom = [], [], [], [], []
    for batch in tqdm(loader, desc=f'{kind} validation inference (flip TTA)', unit='batch', leave=True):
        x = batch['image'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            if kind in ('prototype','trained'):
                a = model(x, return_aux=True)
                b = model(torch.flip(x, dims=[3]), return_aux=True)
                l = (a['linear_logits'] + b['linear_logits']) * .5
                p = (a['proto_logits'] + b['proto_logits']) * .5
                gate = a['prototype_gate']
                lin.append(l.float().cpu().numpy())
                pro.append(p.float().cpu().numpy())
                gates.append(gate.float().cpu().numpy())
            else:
                z = .5*(model(x) + model(torch.flip(x, dims=[3])))
                lin.append(z.float().cpu().numpy())
        y = batch['target'].numpy()
        valid = batch['valid'].numpy()
        ys.append(np.where(valid > 0, y, -1).astype(np.int8))
        dom.append(np.asarray(batch['domain'], dtype=np.int8))
    data = dict(linear=np.concatenate(lin), y=np.concatenate(ys),
                domain=np.concatenate(dom))
    if kind in ('prototype','trained'):
        data.update(proto=np.concatenate(pro), gate=gates[0])
    return data


def per_domain_scores(y, z, d, thresholds):
    p = sigmoid(z)
    overall = evaluate_probs(y, p, thresholds)
    domain = {}
    for index in sorted(int(x) for x in set(d.tolist()) if int(x) >= 0):
        mask = d == index
        if sum(mask) >= 10:
            domain[index] = evaluate_probs(y[mask], p[mask], thresholds)
    scores = [v['challenge_score'] for v in domain.values()]
    # Emphasize consistent quality across the three source domains.
    robust = (.45*overall['challenge_score'] + .30*np.mean(scores) +
              .25*np.min(scores)) if scores else overall['challenge_score']
    return float(robust), overall, domain


def fit_attribute_thresholds(y, z, d):
    """Attribute BA thresholds learned only on the fit subset."""
    p = sigmoid(z)
    grid = np.arange(.30, .76, .025)
    th = np.full(40, .50, dtype=np.float32)
    for c in range(40):
        mask = (y[:, c] >= 0)
        values = y[mask, c]
        if np.count_nonzero(values == 1) < 15 or np.count_nonzero(values == 0) < 15:
            continue
        pp = p[mask, c]
        pos = values == 1
        neg = ~pos
        tpr = (pp[pos, None] >= grid[None, :]).mean(axis=0)
        tnr = (pp[neg, None] < grid[None, :]).mean(axis=0)
        metric = .5*(tpr+tnr)
        # Prefer thresholds near .5 in case of equal balanced accuracy.
        th[c] = float(grid[np.argmax(metric - .0005*np.abs(grid-.5))])
    return th


def operating_options(y, z, d, fit):
    """Very small family; no 40-dimensional unrestricted threshold search."""
    attr = fit_attribute_thresholds(y[fit], z[fit], d[fit])
    for alpha in (0., .20, .40):
        for base in (.48, .50, .52, .55):
            th = np.clip((1-alpha)*base + alpha*attr, .30, .78)
            yield {'alpha':alpha,'base':base}, th.astype(np.float32)


def score_for_choice(y, z, d, mask, threshold):
    robust, met, bydom = per_domain_scores(y[mask], z[mask], d[mask], threshold)
    worst = min((v['challenge_score'] for v in bydom.values()), default=robust)
    # Do not enforce precision == recall; reward the actual challenge components.
    criterion = .70*robust + .30*worst
    return float(criterion), robust, met, bydom


def blend_kind(cache, entry, strength):
    if entry['kind'] in ('prototype','trained'):
        item = cache[entry['key']]
        gate = np.clip(item['gate']*float(strength), 0, .60)
        return ((1-gate)*item['linear'] + gate*item['proto']).astype(np.float32)
    return cache[entry['key']]['linear']


def choices(entries):
    """Small, predeclared candidate family. No per-label gate search."""
    previous = [e for e in entries if e['kind'] == 'prototype']
    trained = [e for e in entries if e['kind'] == 'trained']
    strong = [e for e in entries if e['kind'] == 'strong']
    if not previous or not trained:
        raise RuntimeError('Both original and newly trained Prototype checkpoints are required')

    def pairs(group):
        group = sorted(group, key=lambda x: x['seed'])
        if len(group) >= 2:
            a, b = group[:2]
            for w in ((.60,.40),(.80,.20),(1.,0.),(0.,1.)):
                yield {a['key']:w[0], b['key']:w[1]}
        else:
            yield {group[0]['key']:1.}

    old_pairs=list(pairs(previous))
    new_pairs=list(pairs(trained))
    for s in (1.0, 0.85):
        for w in old_pairs:
            yield {'strength':s,'weights':w,'name':'original_prototype'}
        for w in new_pairs:
            yield {'strength':s,'weights':w,'name':'retrained_prototype'}
        for w_old, w_new in [(old_pairs[0],new_pairs[0])]:
            for frac in (.25,.50,.75):
                weights={k:v*(1-frac) for k,v in w_old.items()}
                for k,v in w_new.items():weights[k]=weights.get(k,0.)+v*frac
                yield {'strength':s,'weights':weights,'name':f'retrained_blend_{frac:.2f}'}
        # optional robust ConvNeXt member, no model-weight explosion
        for strong_model in strong[:1]:
            weights={k:v*.80 for k,v in new_pairs[0].items()}
            weights[strong_model['key']]=.20
            yield {'strength':s,'weights':weights,'name':'trained_plus_strong'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prototype-seeds',default='42,123')
    p.add_argument('--trained-seeds',default='42,123')
    p.add_argument('--strong-seeds',default='42,123')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--force-inference',action='store_true')
    p.add_argument('--anchor-calibration',type=Path,default=None)
    args=p.parse_args()
    seed_everything(2027)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.result_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.repo_root/'data/annotations/task1/val/gt.csv')
    ds=UPARDataset(df,ImageResolver(args.repo_root/'data',args.repo_root),
                   build_eval_transform(288,144))
    fit,audit=stable_split(ds.paths,ds.domains)
    if any(np.sum(audit & (ds.domains==d))<20 for d in (0,1,2)):
        raise RuntimeError('Unknown domain names or insufficient audit data')
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,
                      num_workers=args.num_workers,pin_memory=device.type=='cuda')
    entries=[]
    for kind,seeds in [('prototype',args.prototype_seeds),('trained',args.trained_seeds),('strong',args.strong_seeds)]:
        for seed in [int(x) for x in seeds.split(',') if x.strip()]:
            ckpt=checkpoint_file(args.checkpoint_dir,kind,seed)
            if not ckpt.exists():
                if kind in ('prototype','trained'): raise FileNotFoundError(ckpt)
                print('Optional strong checkpoint unavailable:', ckpt,flush=True)
                continue
            key=f'{kind}:{seed}'
            entries.append(dict(key=key,kind=kind,seed=seed,checkpoint=str(ckpt)))
    cache={}
    for e in entries:
        ckpt=Path(e['checkpoint'])
        key=e['key']
        cache_path=args.result_dir/f"predictions_{e['kind']}_seed{e['seed']}.npz"
        expected=hashlib.sha256((state_signature(ckpt)+'||'+str(args.repo_root)+'||FLIP_288x144').encode()).hexdigest()
        cached=None
        if cache_path.exists() and not args.force_inference:
            with np.load(cache_path) as obj:
                if str(obj['signature'].item())==expected and len(obj['y'])==len(ds):
                    cached={k:obj[k] for k in obj.files if k!='signature'}
        if cached is None:
            ck=load_pt(ckpt)
            state=get_state(ck)
            if e['kind'] in ('prototype','trained'):
                cfg=dict(ck.get('prototype_config',{}));cfg.pop('pretrained',None)
                model=HybridPrototypePAR(pretrained=False,**cfg)
                model.load_state_dict(state,strict=True)
                e['config']=cfg
            else:
                model=StrongPARModel(backbone='convnext_tiny',pretrained=False)
                model.load_state_dict(state,strict=True)
            del ck,state
            model=model.to(device).eval()
            cached=inference(model,e['kind'],loader,device)
            np.savez_compressed(cache_path,signature=expected,**cached)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        else:
            print('Reusing inference cache:',cache_path,flush=True)
        cache[key]=cached
        if e['kind'] in ('prototype','trained') and 'config' not in e:
            e['config']=dict(load_pt(ckpt).get('prototype_config',{}))
        if not np.array_equal(cached['domain'],ds.domains):
            raise RuntimeError('Cached validation domain order mismatch')
    y=next(iter(cache.values()))['y']; d=next(iter(cache.values()))['domain']
    for v in cache.values():
        if not np.array_equal(v['y'],y):raise RuntimeError('Prediction cache label order mismatch')
    print('Domain counts:',{int(i):int((d==i).sum()) for i in np.unique(d)},flush=True)
    print('Fit/audit:',int(fit.sum()),int(audit.sum()),flush=True)

    anchor=None; results=[]
    for candidate in tqdm(list(choices(entries)),desc='Candidate selection on fit only'):
        z=np.zeros((len(y),40),dtype=np.float32)
        for key,w in candidate['weights'].items():
            if w>0:
                ent=next(e for e in entries if e['key']==key)
                z += float(w)*blend_kind(cache,ent,candidate['strength'])
        # Fit threshold and model parameters only on FIT partition.
        best=None
        for thopt, thresholds in operating_options(y,z,d,fit):
            s,robust,met,perdom=score_for_choice(y,z,d,fit,thresholds)
            if best is None or s>best[0]:
                best=(s,robust,met,thopt,thresholds)
        # Immutable audit gets evaluated, not optimized per attribute.
        audit_s,audit_robust,audit_met,audit_domain=score_for_choice(y,z,d,audit,best[4])
        record=dict(name=candidate['name'],strength=candidate['strength'],
                    weights=candidate['weights'],threshold_config=best[3],
                    thresholds=best[4],fit_score=float(best[0]),
                    fit_metrics=best[2],audit_score=float(audit_s),
                    audit_robust=float(audit_robust),audit_metrics=audit_met,
                    audit_by_domain=audit_domain)
        if (candidate['strength']==1.0 and candidate['name']=='original_prototype'
                and len([k for k,v in candidate['weights'].items() if v>0])==2
                and abs(candidate['weights'].get('prototype:42',0)-.6)<.001):
            anchor=record
        results.append(record)
    if anchor is None:
        anchor=next(r for r in results if r['name']=='original_prototype' and r['strength']==1.)
    # Re-evaluate the preserved earlier best prototype calibration as an anchor.
    # This is an out-of-sample *audit* check on the current source validation,
    # not use of any private Codabench labels or hidden scores.
    if args.anchor_calibration and args.anchor_calibration.exists():
        saved = load_pt(args.anchor_calibration)
        try:
            seeds = [int(e['seed']) for e in saved['members']]
            weights = np.asarray(saved['weights'],dtype=np.float64).reshape(-1)
            if len(seeds)==len(weights) and len(set(seeds))==len(seeds):
                reference=np.zeros((len(y),40),dtype=np.float32)
                for seed,w in zip(seeds,weights):
                    key=f'prototype:{seed}'
                    entry=next(e for e in entries if e['key']==key)
                    reference += float(w)*blend_kind(cache,entry,1.0)
                saved_th=sigmoid(np.asarray(saved['threshold_logits'],dtype=np.float64).reshape(40))
                sr,rr,ma,bd = score_for_choice(y,reference,d,audit,saved_th)
                fscore,fr,fmet,fb=score_for_choice(y,reference,d,fit,saved_th)
                refrec=dict(name='original_prototype_calibration',strength=1.0,
                            weights={f'prototype:{seed}':float(w) for seed,w in zip(seeds,weights)},
                            threshold_config={'source':'preserved_original_calibration'},
                            thresholds=saved_th.astype(np.float32),fit_score=fscore,
                            fit_metrics=fmet,audit_score=sr,audit_robust=rr,
                            audit_metrics=ma,audit_by_domain=bd)
                print('Preserved original prototype audit:',ma['challenge_score'],flush=True)
                if refrec['audit_score'] >= anchor['audit_score']:
                    anchor=refrec
        except Exception as exc:
            print('Original calibration unavailable/incompatible:',repr(exc),flush=True)
    results.sort(key=lambda r:r['fit_score'],reverse=True)
    # Strict selection: best FIT candidate, accept only if locked audit beats anchor.
    leader=results[0]
    def weakest(rec):
        arr=[float(m['challenge_score']) for m in rec['audit_by_domain'].values()]
        return min(arr) if arr else 0.
    domain_safe = all(
        int(k) in leader['audit_by_domain']
        and float(leader['audit_by_domain'][int(k)]['challenge_score'])
            >= float(v['challenge_score']) - .002
        for k, v in anchor['audit_by_domain'].items()
    )
    passed=(leader['audit_score'] >= anchor['audit_score']+.003
            and weakest(leader)>=weakest(anchor)-.002
            and domain_safe
            and leader['audit_metrics']['mA']>=anchor['audit_metrics']['mA']-.006
            and leader['audit_metrics']['instance_f1']>=anchor['audit_metrics']['instance_f1']-.006)
    selected=leader if passed else anchor
    print('ANCHOR   :',anchor['name'],round(anchor['fit_score'],5),round(anchor['audit_score'],5),flush=True)
    print('FIT BEST :',leader['name'],round(leader['fit_score'],5),round(leader['audit_score'],5),flush=True)
    print('DECISION :', 'ACCEPT FIT CANDIDATE' if passed else 'FALL BACK TO PROTOTYPE ANCHOR',flush=True)
    print('SELECTED :',selected['weights'], 'strength=',selected['strength'],flush=True)
    print('CONTAINS RETRAINED :',any(k.startswith('trained:') and w>1e-8 for k,w in selected['weights'].items()),flush=True)
    print('AUDIT METRICS:', json.dumps(selected['audit_metrics'],indent=2),flush=True)
    info={
        'format':'upar-generalization-audit',
        'attribute_names':list(ATTRIBUTE_NAMES),
        'entries':entries,
        'weights':selected['weights'],
        'strength':float(selected['strength']),
        'thresholds':torch.tensor(selected['thresholds'],dtype=torch.float32),
        'threshold_logits':torch.tensor(logit(selected['thresholds']),dtype=torch.float32),
        'image_height':288,'image_width':144,'tta':'flip',
        'audit_fraction':.30,
        'audit_metrics':selected['audit_metrics'],
        'fit_metrics':selected['fit_metrics'],
        'anchor_audit':anchor['audit_metrics'],
        'selection_note':'FIT selected / audit-guarded source-domain validation. NOT true unseen-domain evidence',
        'contains_retrained':any(k.startswith('trained:') and w>1e-8 for k,w in selected['weights'].items()),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(info,args.output)
    report={'selected':{k:v for k,v in selected.items() if k!='thresholds'},
            'anchor':{k:v for k,v in anchor.items() if k!='thresholds'},
            'fit_best':{k:v for k,v in leader.items() if k!='thresholds'},
            'top_fit':[{k:v for k,v in r.items() if k not in ('thresholds','audit_by_domain')} for r in results[:8]],
            'accepted':bool(passed),'all_models':[e['key'] for e in entries]}
    report_path=args.result_dir/'generalization_audit_report.json'
    report_path.write_text(json.dumps(report,indent=2,default=float),encoding='utf-8')
    print('SAVED:',args.output, 'REPORT:',report_path,flush=True)

if __name__=='__main__':main()
