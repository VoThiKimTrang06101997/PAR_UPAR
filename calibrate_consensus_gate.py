"""Cross-domain consensus-reliability gate from cached public validation logits.

FIT-only model/threshold selection with a *reused* audit diagnostic. The audit
has been examined previously; it must not be described as a pristine holdout.
No hidden labels, no GPU model training, and no checkpoint modifications.
"""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from calibrate_generalization import stable_split, load_pt, sigmoid, logit
from calibrate_shift_robust import domain_metrics, conservative_thresholds
from calibrate_attribute_adaptive import load_cache, all_logits, weighted_logits
from strong_par import ImageResolver, UPARDataset, build_eval_transform


def consensus_fusion(x:np.ndarray,w:np.ndarray,tau:float,mix:float)->np.ndarray:
    """x (N,K,40), w (K,40). Exact equivalent of bundled runtime logic."""
    anchor=weighted_logits(x,w)
    if tau <= 0. or mix <= 0.:return anchor
    distances=np.abs(sigmoid(x)-sigmoid(anchor)[:,None,:])
    reliability=np.exp(-float(tau)*distances).astype('float32')
    adaptive=reliability*w[None,:,:]
    adaptive/=np.maximum(adaptive.sum(axis=1,keepdims=True),1e-8)
    fused=np.einsum('nkc,nkc->nc',x,adaptive,optimize=True)
    return ((1.-mix)*anchor+mix*fused).astype('float32')


def summary(y,d,mask,z,th):
    score, metrics, per_domain=domain_metrics(y,sigmoid(z),d,mask,th)
    return {'robust':float(score),'metrics':metrics,
            'per_domain':{str(k):v for k,v in per_domain.items()}}


def ranked_gain(new,base):
    metrics=new['metrics'];old=base['metrics']
    gain=float(new['robust']-base['robust'])
    worst=min(new['per_domain'][k]['challenge_score']-base['per_domain'][k]['challenge_score']
              for k in base['per_domain'])
    # Explicitly penalize losses in F1/accuracy/BA and worst domain.
    penalty=(.65*max(0.,old['instance_f1']-metrics['instance_f1'])+
             .20*max(0.,old['Instance_Acc']-metrics['Instance_Acc'])+
             .40*max(0.,old['mA']-metrics['mA'])+
             .30*max(0.,-worst))
    return float(gain-penalty), float(worst)


def run(a):
    a.output.parent.mkdir(parents=True,exist_ok=True)
    reference=load_pt(a.base_calibration)
    if reference.get('format')!='upar-shift-robust-calibration' or reference.get('tta')!='flip':
        raise ValueError('Expected scored AttributeDINO flip calibration')
    entries=[e for e in reference['entries'] if float(reference['weights'].get(e['key'],0.))>1e-7]
    keys=[e['key'] for e in entries]
    weights=np.asarray([float(reference['weights'][key]) for key in keys],dtype='float32')
    weights/=weights.sum()
    w=np.broadcast_to(weights[:,None],(len(keys),40)).copy()
    cache,y,d=load_cache(a.cache_dir,entries,a.repo_root,'flip')
    x=all_logits(reference,cache,entries)
    df=pd.read_csv(a.repo_root/'data/annotations/task1/val/gt.csv')
    ds=UPARDataset(df,ImageResolver(a.repo_root/'data',a.repo_root),build_eval_transform(288,144))
    if len(ds.paths)!=len(y) or not np.array_equal(d,ds.domains) or not np.array_equal(y,np.where(ds.targets>=0,ds.targets,-1)):
        raise RuntimeError('Prediction cache not aligned to official validation labels')
    fit,audit=stable_split(ds.paths,ds.domains)
    hashes=np.asarray([int.from_bytes(hashlib.sha256((str(p)+'::consensus-inner').encode()).digest()[:8], 'little')/2**64 for p in ds.paths])
    inner_train=fit & (hashes>=.25)
    inner_dev=fit & ~inner_train
    for k in (0,1,2):
        assert int((inner_train&(d==k)).sum())>30
        assert int((inner_dev&(d==k)).sum())>30
    threshold_anchor=sigmoid(torch.as_tensor(reference['threshold_logits']).detach().cpu().numpy().reshape(-1)).astype('float32')
    base_z=consensus_fusion(x,w,0.,0.)
    baseline_inner=summary(y,d,inner_dev,base_z,threshold_anchor)
    baseline_fit=summary(y,d,fit,base_z,threshold_anchor)
    baseline_audit=summary(y,d,audit,base_z,threshold_anchor)
    print('[SPLIT] FIT',int(fit.sum()),'inner-train',int(inner_train.sum()),'inner-dev',int(inner_dev.sum()),'reused-audit',int(audit.sum()),flush=True)
    print('[BASE] FIT',baseline_fit['metrics']['challenge_score'],'AUDIT reused',baseline_audit['metrics']['challenge_score'],flush=True)
    # Stable threshold generated exclusively from inner_train labels.
    stable=conservative_thresholds(y,sigmoid(base_z),d,inner_train)
    candidates_th=[('preserved_anchor',threshold_anchor),
                   ('anchor_plus_0.012',np.clip(threshold_anchor+.012,.20,.85)),
                   ('anchor_minus_0.012',np.clip(threshold_anchor-.012,.20,.85)),
                   ('BA_shrink_0.15',np.clip(.85*threshold_anchor+.15*stable,.20,.85))]
    possibilities=[]
    # Hyperparameters specified before looking at reused audit scores.
    for tau in tqdm((0.75,1.5,3.0,6.0),desc='FIT-only consensus candidates',unit='tau'):
        for mix in (.5,1.):
            z=consensus_fusion(x,w,tau,mix)
            for name,th in candidates_th:
                inner=summary(y,d,inner_dev,z,th)
                obj,worst=ranked_gain(inner,baseline_inner)
                possibilities.append(dict(tau=float(tau),mix=float(mix),threshold_name=name,
                                          objective=obj,inner_score=float(inner['metrics']['challenge_score']),
                                          inner_worst_gain=worst))
    choices=sorted(possibilities,key=lambda k:k['objective'],reverse=True)
    choice=choices[0]
    th=dict(candidates_th)[choice['threshold_name']]
    z=consensus_fusion(x,w,choice['tau'],choice['mix'])
    challenger_fit=summary(y,d,fit,z,th)
    # This split was exposed by prior experiments and is ONLY a diagnostic.
    challenger_audit=summary(y,d,audit,z,th)
    obj,worst=ranked_gain(challenger_audit,baseline_audit)
    bm=baseline_audit['metrics'];cm=challenger_audit['metrics']
    gain=float(challenger_audit['robust']-baseline_audit['robust'])
    accepted=(choice['objective']>a.min_inner_gain and gain>=a.min_audit_gain
              and cm['mA']>=bm['mA']-.0015
              and cm['instance_f1']>=bm['instance_f1']-.0015
              and cm['Instance_Acc']>=bm['Instance_Acc']-.0030
              and worst>=-.0020)
    print('[INNER CHOICE]',choice,flush=True)
    print('[REUSED AUDIT]',dict(accepted=accepted,gain=gain,mA_gain=cm['mA']-bm['mA'],F1_gain=cm['instance_f1']-bm['instance_f1'],worst_domain_gain=worst),flush=True)
    def new_cal(tau,mix,threshold,approved):
        c=copy.deepcopy(reference)
        c['format']='upar-consensus-gate-calibration'
        c['attribute_weights']={key:torch.tensor(w[i],dtype=torch.float32) for i,key in enumerate(keys)}
        c['consensus_gate']={'tau':float(tau),'mix':float(mix),'method':'probability_consensus'}
        c['thresholds']=torch.tensor(threshold,dtype=torch.float32)
        c['threshold_logits']=torch.tensor(logit(threshold),dtype=torch.float32)
        c['accepted']=bool(approved)
        c['selection_note']='Public FIT tuning; AUDIT reused diagnostic, not pristine; private unknown'
        return c
    challenger=new_cal(choice['tau'],choice['mix'],th,accepted)
    anchor=new_cal(0.,0.,threshold_anchor,False)
    torch.save(challenger,a.experimental_output)
    torch.save(challenger if accepted else anchor,a.output)
    report=dict(accepted=bool(accepted),selected='consensus_gate' if accepted else 'protected_attribute_dino',
                baseline_private_reference=0.750279,method='consensus_reliability_gate',
                model_keys=keys,inner_fit_choice=choice,inner_fit_ranked_candidates=choices,
                base_fit=baseline_fit,proposed_fit=challenger_fit,
                base_audit=baseline_audit,proposed_audit=challenger_audit,
                audit_reused_not_independent=True,
                gain_robust=float(gain),gain_worst_domain=float(worst),
                diagnostic_ranked_gain=float(obj),
                notes='No private labels; cannot infer private score from source AUDIT; all weights/checkpoints protected.')
    a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[SAVED]',a.report,flush=True)
    print('[SAVED]',a.output,flush=True)
    print('[SAVED]',a.experimental_output,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',required=True,type=Path)
    p.add_argument('--cache-dir',required=True,type=Path)
    p.add_argument('--base-calibration',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--experimental-output',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--min-inner-gain',type=float,default=.00015)
    p.add_argument('--min-audit-gain',type=float,default=.0007)
    args=p.parse_args()
    run(args)
if __name__=='__main__':main()
