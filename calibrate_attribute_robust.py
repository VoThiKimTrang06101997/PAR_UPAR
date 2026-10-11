"""Cross-fitted, group-robust AttributeDINO extension.

All tuning is restricted to source FIT folds. The previously examined source
AUDIT is a post-selection diagnostic (not a new independent holdout).
Private Codabench predictions/labels are never read.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from calibrate_generalization import load_pt, stable_split, sigmoid, logit
from calibrate_shift_robust import domain_metrics, conservative_thresholds
from calibrate_attribute_adaptive import load_cache, all_logits, weighted_logits, audit


def fit_gate(x, y, domains, subset, prior, anchor_threshold,
             penalty=0.8, f1_weight=0.20, steps=90, seed=42):
    """Masked balanced BCE + domain-equal soft instance-F1, L2-to-prior.

    Shapes: x=(N,M,40), y=(N,40) with -1 for unknown, prior=(M,40).
    Only subset rows contribute to gradients.
    """
    torch.manual_seed(seed)
    ids=np.flatnonzero(subset)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xx=torch.from_numpy(np.ascontiguousarray(x[ids])).float().to(device)
    raw=torch.from_numpy(np.ascontiguousarray(y[ids])).float().to(device)
    ds=torch.from_numpy(np.ascontiguousarray(domains[ids])).long().to(device)
    valid=(raw>=0).float()
    target=raw.clamp(0,1)
    prior_t=torch.as_tensor(prior,device=device,dtype=torch.float32)
    threshold_z=torch.as_tensor(logit(anchor_threshold),device=device,dtype=torch.float32).view(1,-1)
    balanced=torch.zeros_like(raw)
    for domain in (0,1,2):
        dom_mask=(ds==domain).float().unsqueeze(1)
        for cls in (0.,1.):
            selection=dom_mask*valid*(target==cls).float()
            count=selection.sum(0)
            balanced+=selection*torch.where(count>=10,1./count.clamp_min(1),0.).view(1,-1)
    balanced=balanced/balanced.sum().clamp_min(1.e-8)
    delta=torch.nn.Parameter(torch.zeros_like(prior_t))
    opt=torch.optim.Adam([delta],lr=.045)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        weights=torch.softmax(torch.log(prior_t.clamp_min(1e-5))+delta,dim=0)
        zz=torch.einsum('nmc,mc->nc',xx,weights)
        ce=(F.binary_cross_entropy_with_logits(zz,target,reduction='none')*balanced).sum()
        # A differentiable proxy only: final selection uses exact evaluate_probs.
        probabilities=torch.sigmoid(1.7*(zz-threshold_z))
        per_domain_f1=[]
        for dom in (0,1,2):
            sub=(ds==dom)
            if sub.sum()<10:continue
            p=probabilities[sub]*valid[sub]
            yy=target[sub]*valid[sub]
            tp=(p*yy).sum(1)
            union=p.sum(1)+yy.sum(1)
            relevant=(valid[sub].sum(1)>0)&(union>1e-5)
            if relevant.any():
                per_domain_f1.append((2*tp[relevant]/union[relevant].clamp_min(1e-5)).mean())
        soft_f1=(torch.stack(per_domain_f1).mean() if per_domain_f1 else zz.new_tensor(0.))
        reg=((weights-prior_t)**2).sum()/40.
        loss=ce+float(f1_weight)*(1.-soft_f1)+float(penalty)*reg
        if not torch.isfinite(loss): raise RuntimeError('Non-finite calibration loss')
        loss.backward()
        opt.step()
    return torch.softmax(torch.log(prior_t.clamp_min(1e-5))+delta,dim=0).detach().cpu().numpy().astype('float32')


def make_prior(anchor, extra_index, extra_fraction):
    prior=anchor.copy()
    if extra_index is not None and extra_fraction>0:
        prior*=1.-extra_fraction
        prior[extra_index,:]=extra_fraction
    return prior/prior.sum(axis=0,keepdims=True)


def threshold_choice(y,z,domain,train_mask,anchor,variant):
    if variant=='anchor':return anchor.copy()
    if variant not in ('stable_25','stable_50'):raise KeyError(variant)
    stable=conservative_thresholds(y,sigmoid(z),domain,train_mask)
    alpha=.25 if variant=='stable_25' else .50
    return np.clip((1-alpha)*anchor+alpha*stable,.25,.80).astype('float32')


def metric(y,d,mask,z,th):
    robust,summary,perdomain=domain_metrics(y,sigmoid(z),d,mask,th)
    return {'robust':float(robust),'summary':summary,'per_domain':perdomain}


def fold_ids(paths,mask,domains,nfold=3):
    # Deterministic, domain-balanced fold allocation, independent of target labels.
    result=np.full(len(paths),-1,dtype='int16')
    for dom in (0,1,2):
        inds=np.flatnonzero(mask & (domains==dom))
        ordering=sorted(inds,key=lambda i: hashlib.sha256((str(paths[i]).lower()+'|robust-fold').encode()).digest())
        for rank,index in enumerate(ordering):result[index]=rank%nfold
    if np.any(mask & (result<0)):raise RuntimeError('Fold creation failed')
    return result


def score_candidate(candidate,baseline):
    a=candidate['summary'];b=baseline['summary']
    dm=float(a['mA']-b['mA'])
    df=float(a['instance_f1']-b['instance_f1'])
    dr=float(candidate['robust']-baseline['robust'])
    dd=min(float(candidate['per_domain'][k]['challenge_score']-baseline['per_domain'][k]['challenge_score']) for k in (0,1,2))
    # Guard against pursuing mA alone: penalize F1 and weakest-domain regressions.
    utility=(dr + .70*min(0.,df+.0005) + .50*min(0.,dm+.0005) + .40*min(0.,dd+.002))
    return {'utility':utility,'robust_delta':dr,'mA_delta':dm,'instance_f1_delta':df,'worst_domain_delta':dd}


def save_json(path, record):
    def cast(z):
        if isinstance(z,np.generic):return z.item()
        if isinstance(z,np.ndarray):return z.tolist()
        raise TypeError(type(z).__name__)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2,default=cast),encoding='utf-8')


def run(args):
    from strong_par import ImageResolver, UPARDataset, build_eval_transform
    import pandas as pd
    base=load_pt(args.base_calibration)
    expanded=load_pt(args.expanded_calibration)
    if base.get('format')!='upar-shift-robust-calibration':
        raise ValueError('Expected frozen original AttributeDINO calibration')
    if base.get('tta')!='flip' or expanded.get('tta')!='flip':
        raise ValueError('Flip-TTA required for cache alignment')
    base_entries=[e for e in base['entries'] if float(base['weights'].get(e['key'],0))>1e-7]
    entries=[e for e in expanded['entries'] if float(expanded['weights'].get(e['key'],0))>1e-7]
    names=[e['key'] for e in entries]
    if len(names)!=len(set(names)):raise RuntimeError('Duplicate model keys')
    base_names={e['key'] for e in base_entries}
    if not base_names.issubset(names):raise RuntimeError('Expanded calibration lost a baseline member')
    optional=[i for i,k in enumerate(names) if k not in base_names]
    if len(optional)>1:raise RuntimeError('Only one independent added model is supported')
    extra=optional[0] if optional else None
    print('[MODELS]',names,'extra=',names[extra] if extra is not None else None,flush=True)
    cache,y,domains=load_cache(args.result_dir,entries,args.repo_root,'flip')
    x=all_logits(expanded,cache,entries)
    data=pd.read_csv(args.repo_root/'data/annotations/task1/val/gt.csv')
    dataset=UPARDataset(data,ImageResolver(args.repo_root/'data',args.repo_root),build_eval_transform(288,144))
    if len(dataset)!=len(y) or not np.array_equal(dataset.domains,domains) or not np.array_equal(np.where(dataset.targets>=0,dataset.targets,-1),y):
        raise RuntimeError('Cached targets/domains do not match official validation ordering')
    fit,audit_mask=stable_split(dataset.paths,domains)
    fold=fold_ids(dataset.paths,fit,domains)
    anchor_threshold=sigmoid(torch.as_tensor(base['threshold_logits']).cpu().numpy().reshape(-1)).astype('float32')
    base_weights=np.asarray([float(base['weights'].get(k,0.)) for k in names],dtype='float32')
    base_weights/=base_weights.sum()
    anchor=np.repeat(base_weights[:,None],40,axis=1)
    base_z=weighted_logits(x,anchor)
    fractions=[0.] if extra is None else [0.,.15,.30]
    # MODEL SELECTION ONLY FROM THREE INNER-FIT FOLDS.
    candidates=[]
    options=[(p,q,f,s,t) for p in (.4,1.2) for q in (.10,.30)
             for f in fractions for s in (.35,.65) for t in ('anchor','stable_25')]
    print('[INNER FIT] candidates=',len(options),'folds=3',flush=True)
    # Cache fitted fold gates so changing threshold or shrink does not refit models.
    fitted={}
    for pen in (.4,1.2):
        for f1 in (.10,.30):
            for frac in fractions:
                prior=make_prior(anchor,extra,frac)
                for k in range(3):
                    train=fit&(fold!=k)
                    key=(pen,f1,frac,k)
                    learned=fit_gate(x,y,domains,train,prior,anchor_threshold,
                                     penalty=pen,f1_weight=f1,steps=args.steps,seed=42+k)
                    fitted[key]=learned
                print(f'[INNER FIT] penalty={pen} f1={f1} extra_prior={frac}: trained 3 folds',flush=True)
    for pen,f1,frac,shrink,thtype in tqdm(options,desc='Cross-validated candidate selection',unit='candidate'):
        scores=[]
        prior=make_prior(anchor,extra,frac)
        for k in range(3):
            train=fit&(fold!=k);dev=fit&(fold==k)
            learned=fitted[(pen,f1,frac,k)]
            weights=((1.-shrink)*anchor+shrink*learned).astype('float32')
            weights/=weights.sum(0,keepdims=True)
            z=weighted_logits(x,weights)
            th=threshold_choice(y,z,domains,train,anchor_threshold,thtype)
            candidate=metric(y,domains,dev,z,th)
            baseline=metric(y,domains,dev,base_z,anchor_threshold)
            scores.append(score_candidate(candidate,baseline))
        mean={key:float(np.mean([row[key] for row in scores])) for key in scores[0]}
        mean['positive_folds']=int(sum(s['robust_delta']>0 for s in scores))
        record=dict(penalty=pen,f1_weight=f1,extra_fraction=frac,shrink=shrink,
                    threshold_family=thtype,**mean)
        candidates.append(record)
    candidates.sort(key=lambda row:row['utility'],reverse=True)
    winner=candidates[0]
    print('[INNER FIT SELECTED]',winner,flush=True)
    # Refit on ALL FIT after all hyperparameters are frozen. AUDIT unseen to optimizer.
    prior=make_prior(anchor,extra,winner['extra_fraction'])
    learned=fit_gate(x,y,domains,fit,prior,anchor_threshold,penalty=winner['penalty'],
                     f1_weight=winner['f1_weight'],steps=args.steps)
    weights=(1-winner['shrink'])*anchor+winner['shrink']*learned
    weights=(weights/weights.sum(0,keepdims=True)).astype('float32')
    znew=weighted_logits(x,weights)
    th=threshold_choice(y,znew,domains,fit,anchor_threshold,winner['threshold_family'])
    reference_fit=audit(y,domains,fit,base_z,anchor_threshold)
    proposed_fit=audit(y,domains,fit,znew,th)
    reference_audit=audit(y,domains,audit_mask,base_z,anchor_threshold)
    proposed_audit=audit(y,domains,audit_mask,znew,th)
    source_audit_change=score_candidate({'robust':proposed_audit['robust'],'summary':proposed_audit['metrics'],
                                         'per_domain':{int(k):v for k,v in proposed_audit['per_domain'].items()}},
                                        {'robust':reference_audit['robust'],'summary':reference_audit['metrics'],
                                         'per_domain':{int(k):v for k,v in reference_audit['per_domain'].items()}})
    cv_positive=(winner['robust_delta']>0.0005 and winner['positive_folds']>=2 and winner['instance_f1_delta']>=-.0015)
    accepted=bool(cv_positive and source_audit_change['robust_delta']>=.001 and
                  source_audit_change['instance_f1_delta']>=-.001 and
                  source_audit_change['mA_delta']>=-.001 and
                  source_audit_change['worst_domain_delta']>=-.003)
    print('[SOURCE AUDIT DIAGNOSTIC]',source_audit_change,'accepted=',accepted,flush=True)
    # Original protected reference is never overwritten.
    chosen=copy.deepcopy(base)
    chosen['format']='upar-attribute-adaptive-calibration'
    chosen['attribute_weights']={e['key']:torch.as_tensor(anchor[i].copy()) for i,e in enumerate(entries) if e['key'] in base_names}
    chosen['accepted']=False
    chosen['selection_note']='Protected previously scored AttributeDINO anchor; unchanged predictions.'
    cand=copy.deepcopy(expanded)
    cand['format']='upar-attribute-adaptive-calibration'
    cand['attribute_weights']={k:torch.as_tensor(weights[i].copy()) for i,k in enumerate(names)}
    cand['threshold_logits']=torch.as_tensor(logit(th).astype('float32'))
    cand['thresholds']=torch.as_tensor(th)
    cand['accepted']=accepted
    cand['audit_metrics']=proposed_audit['metrics']
    cand['selection_note']='Cross-fitted FIT-only selection and one post-hoc source-AUDIT diagnostic. Codabench score unknown.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cand,args.experimental_output)
    torch.save(cand if accepted else chosen,args.output)
    report={'accepted':accepted,'selected':'crossfit_metric_aware' if accepted else 'protected_attribute_dino',
            'model_keys':names,'optional_added_model':names[extra] if extra is not None else None,
            'fit_rows':int(fit.sum()),'audit_rows':int(audit_mask.sum()),
            'cross_validation':{'best':winner,'top5':candidates[:5],'tested_candidate_count':len(candidates)},
            'reference_fit':reference_fit,'candidate_fit':proposed_fit,
            'reference_audit':reference_audit,'candidate_audit':proposed_audit,
            'posthoc_audit_delta':source_audit_change,
            'attribute_max_change':float(np.abs(weights-anchor).max()),
            'attribute_mean_change':float(np.abs(weights-anchor).mean()),
            'outputs':{'audit_selected':str(args.output),'experimental':str(args.experimental_output)},
            'warning':'Source AUDIT has been used in prior experiments and is NOT a pristine independent test. Hidden/private score not predicted.'}
    save_json(args.report,report)
    print('[SAVED]',args.output,args.experimental_output,args.report,flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ('repo-root','result-dir','base-calibration','expanded-calibration','output','experimental-output','report'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--steps',type=int,default=90)
    args=p.parse_args()
    for path in (args.base_calibration,args.expanded_calibration):
        if not path.is_file():raise FileNotFoundError(path)
    run(args)

if __name__=='__main__':main()
