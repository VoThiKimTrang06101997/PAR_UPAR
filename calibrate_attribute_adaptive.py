"""Train FIT-only attribute-wise ensemble weights using cached source validation logits.

A locked AUDIT split is accessed exactly once after FIT hyperparameter selection.
No private/hidden labels. Original model checkpoints are NEVER modified.
"""
from __future__ import annotations
import argparse, copy, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from calibrate_generalization import stable_split, load_pt, get_state, sigmoid, logit
from calibrate_shift_robust import signature, domain_metrics, conservative_thresholds


def load_cache(base, entries, repo_root, mode):
    cache={}; y=None; domain=None
    for e in entries:
        cache_file=base/f"shift_predictions_{e['kind']}_seed{e['seed']}_{mode}.npz"
        if not cache_file.is_file():
            raise FileNotFoundError(f'Missing GPU prediction cache: {cache_file}. Run original Cell 9 first.')
        with np.load(cache_file, allow_pickle=False) as record:
            want=signature(Path(e['checkpoint']),mode,repo_root)
            got=str(record['signature'].item())
            if got!=want:
                raise RuntimeError(f'Stale cache: {cache_file}. Re-run original source calibration to refresh.')
            item={k:record[k].copy() for k in ('linear','proto','y','domain')}
        if y is None:
            y=item['y']; domain=item['domain']
        elif not np.array_equal(y,item['y']) or not np.array_equal(domain,item['domain']):
            raise RuntimeError(f'Cache row mismatch for {e["key"]}')
        cache[e['key']]=item
        print('[CACHE PASS]', e['key'], item['linear'].shape, flush=True)
    if y.shape[1]!=40 or set(np.unique(domain).tolist())!={0,1,2}:
        raise ValueError('Expected (N,40) source validation with three domains')
    return cache,y.astype('int8'),domain.astype('int8')


def gate_from_checkpoint(entry):
    if entry['kind']=='strong':return np.zeros((1,40),dtype='float32')
    ck=load_pt(Path(entry['checkpoint']))
    sd=get_state(ck)
    x=sd['prototype_gate_logits'].detach().float().cpu().numpy().reshape(1,40)
    config=dict(entry.get('config') or {})
    lo= .05 if entry['kind']=='dinov2' else float(config.get('prototype_gate_min',.08))
    hi= .65 if entry['kind']=='dinov2' else float(config.get('prototype_gate_max',.92))
    return np.clip(sigmoid(x),lo,hi)


def all_logits(cal,cache,entries):
    strength=float(cal['strength']); out=[]
    for e in entries:
        record=cache[e['key']]
        gate=np.clip(gate_from_checkpoint(e)*strength,0.,.60)
        logits=(1.-gate)*record['linear']+gate*record['proto']
        if not np.isfinite(logits).all():raise ValueError(f'NaN logits from {e["key"]}')
        out.append(logits.astype('float32'))
    return np.stack(out,axis=1)


def weighted_logits(x, w):
    # x (N,K,C), w (K,C); same algebra used by Codabench runtime.
    return np.einsum('nkc,kc->nc',x,w,optimize=True).astype('float32')


def fit_softmax_weights(x, y, d, train_mask, reference, penalty=.5, steps=120, seed=42):
    """Masked balanced BCE pooled over domains and classes; regularized toward reference.

    Optimization learns a LOW-dimensional Kx40 gate only, never updates image models.
    Training examples are drawn exclusively from FIT train_mask.
    """
    torch.manual_seed(seed)
    idx=np.flatnonzero(train_mask)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx=torch.from_numpy(np.ascontiguousarray(x[idx])).float().to(device)
    yy=torch.from_numpy(np.ascontiguousarray(y[idx])).float().to(device)
    dd=torch.from_numpy(np.ascontiguousarray(d[idx])).long().to(device)
    known=(yy>=0).float(); yy=yy.clamp(0.,1.)
    weights=torch.zeros_like(yy)
    for dom in (0,1,2):
        in_domain=(dd==dom).float().unsqueeze(1)
        for label in (0.,1.):
            ok=in_domain*known*(yy==label).float()
            counts=ok.sum(0)
            # Both classes required for balanced statistics; otherwise skip stratum.
            weights+=ok*torch.where(counts>=10,1./counts.clamp_min(1.),0.).view(1,-1)
    sums=weights.sum(0).clamp_min(1e-6)
    weights=weights/sums
    base=torch.tensor(reference,dtype=torch.float32,device=device)
    delta=torch.nn.Parameter(torch.zeros_like(base))
    opt=torch.optim.Adam([delta],lr=.065)
    # Limited steps, full training FIT set; no stochastic hyperparameter dependence.
    for step in tqdm(range(steps),desc=f'FIT attribute weights λ={penalty}',unit='step'):
        opt.zero_grad(set_to_none=True)
        adapted=torch.softmax(torch.log(base.clamp_min(.0001))+delta,dim=0)
        zz=torch.einsum('nkc,kc->nc',xx,adapted)
        ce=F.binary_cross_entropy_with_logits(zz,yy,reduction='none')
        loss=(ce*weights).sum()/40.
        # Hard shrink away from unstable per-attribute weights.
        regularizer=((adapted-base)**2).sum()/40.
        loss=(loss+float(penalty)*regularizer)
        loss.backward();opt.step()
    learned=torch.softmax(torch.log(base.clamp_min(.0001))+delta,dim=0).detach().cpu().numpy()
    return learned.astype('float32')


def thresholds(y,prob,domain,fit_mask,anchor_th):
    stable=conservative_thresholds(y,prob,domain,fit_mask)
    # Keep a short, stable threshold family. No per-attribute search on AUDIT.
    choices=[('anchor',anchor_th)]
    for blend in (.35,.65):
        for center in (.45,.50,.55):
            t=np.clip((1.-blend)*center+blend*stable,.25,.80)
            choices.append((f'BA{blend:.2f}_center{center:.2f}',t.astype('float32')))
    return choices


def metric(y,d,valid,z,th):
    return domain_metrics(y,sigmoid(z),d,valid,th)


def tune_threshold(y,d,mask,z,anchor_th):
    best=None
    for name, th in thresholds(y,sigmoid(z),d,mask,anchor_th):
        score,metrics,perdomain=metric(y,d,mask,z,th)
        if best is None or score>best[0]:
            best=(score,name,th,metrics,perdomain)
    return best


def audit(y,d,mask,z,th):
    robust,metrics,doms=metric(y,d,mask,z,th)
    return dict(robust=float(robust),metrics=metrics,
                per_domain={str(k):v for k,v in doms.items()})


def run(args):
    args.result_dir.mkdir(parents=True,exist_ok=True)
    cal=load_pt(args.base_calibration)
    if cal.get('format')!='upar-shift-robust-calibration':
        raise ValueError('Expected the previous AttributeDINO shift-robust calibration')
    if cal.get('tta')!='flip':
        raise ValueError('This calibration supports flip TTA only; re-run original Cell 9 with flip')
    entries=[e for e in cal['entries'] if float(cal['weights'].get(e['key'],0.))>1e-7]
    names=[e['key'] for e in entries]
    if len(entries)<2:raise RuntimeError('At least two active models required')
    base=np.asarray([float(cal['weights'][k]) for k in names],dtype='float32')
    base=base/base.sum()
    reference=np.tile(base[:,None],(1,40)).astype('float32')
    print('[MODEL KEYS]', names, '[GLOBAL]',base.tolist(),flush=True)
    cache,y,d=load_cache(args.result_dir,entries,args.repo_root,'flip')
    x=all_logits(cal,cache,entries)
    from strong_par import ImageResolver
    import pandas as pd
    df=pd.read_csv(args.repo_root/'data/annotations/task1/val/gt.csv')
    image_col='# image' if '# image' in df else 'image'
    if image_col not in df:raise RuntimeError('Missing official image column')
    if len(df)!=len(y):raise RuntimeError('Cache count vs annotation mismatch')
    # Match the original calibration's exact resolved-path split.
    # Reconstruct exact split keys with ImageResolver, which may cost ~2s on local SSD.
    from strong_par import UPARDataset, build_eval_transform
    ds=UPARDataset(df,ImageResolver(args.repo_root/'data',args.repo_root),build_eval_transform(288,144))
    fit,audit_mask=stable_split(ds.paths,ds.domains)
    if not np.array_equal(d,ds.domains):raise RuntimeError('Cache/official source domain mismatch')
    if not np.array_equal(y,np.where(ds.targets>=0,ds.targets,-1)):
        raise RuntimeError('Cache/official source labels mismatch')
    # Split FIT deterministically into inner-train and inner-dev, but leave AUDIT untouched.
    hashes=np.array([int.from_bytes(hashlib.sha256((str(path)+'::gate-tune').encode()).digest()[:8],'little')/2**64
                     for path in ds.paths])
    inner_train=fit & (hashes>=.22)
    inner_dev=fit & ~inner_train
    for dom in (0,1,2):
        if (inner_dev & (d==dom)).sum()<20: raise RuntimeError('Insufficient inner-dev images')
    threshold_anchor=sigmoid(torch.as_tensor(cal['threshold_logits']).cpu().numpy().reshape(-1)).astype('float32')
    zbase=weighted_logits(x,reference)
    base_fit=audit(y,d,fit,zbase,threshold_anchor)
    base_locked=audit(y,d,audit_mask,zbase,threshold_anchor)
    print('[SOURCE BASE] FIT=',base_fit['metrics']['challenge_score'], 'AUDIT=',base_locked['metrics']['challenge_score'],flush=True)
    candidates=[]
    # Fit-internal model selection; do not touch locked AUDIT labels here.
    for penalty in (.2, .8, 2.):
        learned=fit_softmax_weights(x,y,d,inner_train,reference,penalty=penalty,steps=args.steps)
        for shift in (.30,.55,.80):
            ww=(1.-shift)*reference+shift*learned
            zz=weighted_logits(x,ww)
            fitted=tune_threshold(y,d,inner_train,zz,threshold_anchor)
            inner=audit(y,d,inner_dev,zz,fitted[2])
            inner_baseline=audit(y,d,inner_dev,zbase,threshold_anchor)
            domains=inner['per_domain']; base_domains=inner_baseline['per_domain']
            minimum=min(float(domains[k]['challenge_score']-base_domains[k]['challenge_score']) for k in domains)
            objective=(inner['robust']-inner_baseline['robust'])+.20*min(0.,minimum)
            candidates.append(dict(penalty=penalty,shift=shift,inner_gain=float(objective),
                                   inner_source_score=float(inner['metrics']['challenge_score']),
                                   threshold_name=fitted[1]))
    ordered=sorted(candidates,key=lambda z:z['inner_gain'],reverse=True)
    selected=ordered[0]
    print('[FIT INTERNAL CHOICE]',selected,flush=True)
    learned=fit_softmax_weights(x,y,d,fit,reference,penalty=selected['penalty'],steps=args.steps)
    weights=((1.-selected['shift'])*reference+selected['shift']*learned).astype('float32')
    weights/=weights.sum(0,keepdims=True)
    new_z=weighted_logits(x,weights)
    tfit=tune_threshold(y,d,fit,new_z,threshold_anchor)
    proposed_fit=audit(y,d,fit,new_z,tfit[2])
    proposed_locked=audit(y,d,audit_mask,new_z,tfit[2])
    bm=base_locked['metrics'];nm=proposed_locked['metrics']
    bdomains=base_locked['per_domain']; ndomains=proposed_locked['per_domain']
    diff=float(proposed_locked['robust']-base_locked['robust'])
    accepted=(diff>=args.min_audit_gain and
              nm['mA']>=bm['mA']-.002 and
              nm['instance_f1']>=bm['instance_f1']-.002 and
              all(ndomains[k]['challenge_score']>=bdomains[k]['challenge_score']-.003 for k in bdomains))
    print('[LOCKED AUDIT]',{'gain':diff,'accepted':accepted,'mA':nm['mA'],'instance_f1':nm['instance_f1']},flush=True)
    # Save both: accepted-only for safe choice, and explicitly experimental for comparison.
    candidate=copy.deepcopy(cal)
    candidate['format']='upar-attribute-adaptive-calibration'
    candidate['attribute_weights']={key:torch.from_numpy(weights[j].copy()) for j,key in enumerate(names)}
    candidate['thresholds']=torch.from_numpy(np.asarray(tfit[2],dtype='float32').copy())
    candidate['threshold_logits']=torch.from_numpy(logit(np.asarray(tfit[2],dtype='float32')).copy())
    candidate['audit_metrics']=nm
    candidate['accepted']=bool(accepted)
    candidate['selection_note']='FIT-only attribute weight learning; inner-FIT tuning; locked AUDIT evaluated once. Private score unknown.'
    anchor=copy.deepcopy(cal)
    anchor['format']='upar-attribute-adaptive-calibration'
    anchor['attribute_weights']={key:torch.from_numpy(reference[j].copy()) for j,key in enumerate(names)}
    anchor['accepted']=False
    anchor['selection_note']='Protected AttributeDINO base mixture; unchanged predictions.'
    args.experimental_output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(candidate,args.experimental_output)
    chosen=candidate if accepted else anchor
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(chosen,args.output)
    changes=np.abs(weights-reference)
    report=dict(accepted=accepted,selected=('attribute_adaptive' if accepted else 'protected_attribute_dino'),
                reference_models=names,reference_weights=base.tolist(),
                attribute_weight_max_change=float(changes.max()),attribute_weight_mean_change=float(changes.mean()),
                fit_count=int(fit.sum()),inner_dev_count=int(inner_dev.sum()),audit_count=int(audit_mask.sum()),
                base_fit=base_fit,proposed_fit=proposed_fit,base_audit=base_locked,proposed_audit=proposed_locked,
                locked_gain_robust=diff,threshold_name=tfit[1],inner_fit_candidates=ordered,
                audit_calibration=str(args.output),experiment_calibration=str(args.experimental_output),
                note='All metrics SOURCE validation only; no claim private >0.78.')
    args.report.write_text(json.dumps(report,indent=2,default=lambda o:float(o) if isinstance(o,np.number) else o))
    print('[SAVED] audit-selected',args.output,flush=True)
    print('[SAVED] distinct experimental',args.experimental_output,flush=True)
    print('[SAVED] report',args.report,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--base-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experimental-output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--steps',type=int,default=95)
    p.add_argument('--min-audit-gain',type=float,default=.001)
    args=p.parse_args()
    for path in (args.base_calibration,):
        if not path.is_file():raise FileNotFoundError(path)
    run(args)

if __name__=='__main__':main()
