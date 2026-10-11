"""Domain-balanced DINOv2-S + attribute queries, masked ASL, consistency, SupCon.

Unseen/private labels are never touched. All progress is reported to Colab.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from strong_par import (ATTRIBUTE_NAMES, ImageResolver, UPARDataset,
                        build_train_transform, build_eval_transform, dataset_pos_weight,
                        seed_everything)
from domain_contrastive import (DomainMixBatchSampler, cross_domain_supcon,
                                domain_positive_negative_alignment)
from dinov2_par import DinoAttributePAR, config_for_checkpoint
from prototype_par import HybridPrototypePAR
from calibrate_generalization import stable_split
from vapor_par.metrics import evaluate_probs


def masked_asl(logits, targets, valid, pos_weights, gamma_negative=3., gamma_positive=0., neg_clip=.04):
    """Masked multi-label ASL: easy negative suppression + bounded attribute priors."""
    p=torch.sigmoid(logits.float()).clamp(1e-6,1-1e-6)
    negative=(1.-p+neg_clip).clamp(max=1.)
    positive_term=targets*torch.log(p)*torch.pow(1.-p,gamma_positive)*pos_weights.view(1,-1)
    negative_term=(1-targets)*torch.log(negative)*torch.pow(1.-negative,gamma_negative)
    loss=-(positive_term+negative_term)*valid
    return loss.sum()/valid.sum().clamp_min(1.)


def prototype_separation(aux):
    return torch.relu((aux['prototype_pos']*aux['prototype_neg']).sum(-1)+.15).mean()


def mild_domain_augmentation(x):
    """Appearance shifts with no label-changing crop or occlusion."""
    x=x.float()
    # Perturb normalized tensor only mildly to preserve clothing-color labels.
    strength=torch.empty((x.shape[0],1,1,1),device=x.device).uniform_(.96,1.04)
    offset=torch.empty((x.shape[0],1,1,1),device=x.device).uniform_(-.06,.06)
    out=(x*strength+offset).clamp(-3.,3.)
    if torch.rand((),device=x.device)<.5:
        out=out.flip(-1)
    return out


def checkpoint_paths(root,seed,heldout,encoder):
    suffix='' if heldout<0 else f'_heldout{heldout}'
    stem=(f'dinov2_attribute_seed{seed}{suffix}' if encoder=='dinov2'
          else f'attribute_robust_convnext_seed{seed}{suffix}')
    return root/f'{stem}_last.pt',root/f'{stem}_best.pt'


def _read(path):
    return torch.load(path,map_location='cpu',weights_only=False)


@torch.inference_mode()
def evaluate(model,loader,device,domain_filter=None):
    model.eval()
    probs=[]; targets=[]; domains=[]
    for batch in tqdm(loader,desc='DINO source FIT validation',unit='batch',leave=False,dynamic_ncols=True):
        x=batch['image'].to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            logits=(model(x).float()+model(x.flip(-1)).float())*.5
        probs.append(torch.sigmoid(logits).cpu().numpy())
        raw=batch['target'].numpy(); v=batch['valid'].numpy()
        targets.append(np.where(v>0,raw,-1))
        domains.append(np.asarray(batch['domain']))
    y=np.concatenate(targets);p=np.concatenate(probs);d=np.concatenate(domains)
    metrics=evaluate_probs(y,p,np.full(40,.5,dtype='float32'))
    per_domain={}
    for domain in (0,1,2):
        ix=d==domain
        if ix.sum()>5:
            per_domain[str(domain)]=evaluate_probs(y[ix],p[ix],np.full(40,.5,dtype='float32'))
    challenge=[v['challenge_score'] for v in per_domain.values()]
    worst=min(challenge) if challenge else metrics['challenge_score']
    objective=.50*float(metrics['challenge_score'])+.25*float(np.mean(challenge))+.25*float(worst)
    model.train()
    return objective,metrics,per_domain


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--encoder',choices=['dinov2','convnext'],default='dinov2')
    p.add_argument('--init-prototype-checkpoint',type=Path,default=None)
    p.add_argument('--heldout-domain',type=int,default=-1)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--steps-per-epoch',type=int,default=350)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--val-limit',type=int,default=3600)
    p.add_argument('--sampler-alpha',type=float,default=.65)
    p.add_argument('--trainable-last',type=int,default=2)
    p.add_argument('--head-lr',type=float,default=2.5e-4)
    p.add_argument('--backbone-lr',type=float,default=8e-6)
    p.add_argument('--weight-decay',type=float,default=.025)
    p.add_argument('--asl-weight',type=float,default=1.)
    p.add_argument('--supcon-weight',type=float,default=.02)
    p.add_argument('--alignment-weight',type=float,default=.008)
    p.add_argument('--consistency-weight',type=float,default=.035)
    p.add_argument('--separation-weight',type=float,default=.005)
    p.add_argument('--warmup-epochs',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not (-1<=args.heldout_domain<=2): raise ValueError('Invalid domain')
    if args.steps_per_epoch<1 or args.epochs<1:raise ValueError('Invalid epochs/steps')
    if args.batch_size<9:raise ValueError('batch-size should be >=9 to sample 3 source domains')
    seed_everything(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    last_path,best_path=checkpoint_paths(args.checkpoint_dir,args.seed,args.heldout_domain,args.encoder)
    print('[DATA] Reading official source train/val (NO private data)',flush=True)
    train_df=pd.read_csv(args.repo_root/'data/annotations/task1/train/gt.csv')
    val_df=pd.read_csv(args.repo_root/'data/annotations/task1/val/gt.csv')
    resolver=ImageResolver(args.repo_root/'data',args.repo_root)
    height,width=(252,126) if args.encoder=='dinov2' else (288,144)
    train_ds=UPARDataset(train_df,resolver,build_train_transform(height,width))
    val_ds=UPARDataset(val_df,resolver,build_eval_transform(height,width))
    train_ids=np.flatnonzero((train_ds.domains!=args.heldout_domain) if args.heldout_domain>=0 else (train_ds.domains>=0))
    if not len(train_ids):raise RuntimeError('No valid train domains')
    fit,audit=stable_split(val_ds.paths,val_ds.domains)
    selection_mask=fit & ((val_ds.domains!=args.heldout_domain) if args.heldout_domain>=0 else (val_ds.domains>=0))
    val_ids=np.flatnonzero(selection_mask)
    rng=np.random.default_rng(args.seed+41)
    if len(val_ids)>args.val_limit:
        # Every observed source domain receives a similar minimum representation.
        per=max(1,args.val_limit//len(np.unique(val_ds.domains[val_ids])))
        picked=[]
        for domain in np.unique(val_ds.domains[val_ids]):
            sub=val_ids[val_ds.domains[val_ids]==domain]
            picked.extend(rng.choice(sub,size=min(per,len(sub)),replace=False).tolist())
        val_ids=np.asarray(sorted(picked),dtype='int64')
    if not len(val_ids):raise RuntimeError('No source FIT validation examples')
    # Sampler indexes are subset-local; prevent held-out label leakage.
    train_loader=DataLoader(Subset(train_ds,train_ids.tolist()),
        batch_sampler=DomainMixBatchSampler(train_ds.domains[train_ids],
        batch_size=args.batch_size,steps_per_epoch=args.steps_per_epoch,alpha=args.sampler_alpha,
        seed=args.seed),num_workers=args.num_workers,pin_memory=device.type=='cuda')
    val_loader=DataLoader(Subset(val_ds,val_ids.tolist()),batch_size=max(1,args.batch_size),
        shuffle=False,num_workers=args.num_workers,pin_memory=device.type=='cuda')
    pos=dataset_pos_weight(train_ds.targets[train_ids],exponent=.30,max_weight=3.).to(device)
    print('[DATA] train=',len(train_ids),'val-FIT=',len(val_ids),'domains=',np.unique(train_ds.domains[train_ids]).tolist(),flush=True)
    if args.encoder=='dinov2':
        model=DinoAttributePAR(trainable_last=args.trainable_last)
    else:
        ck_origin=None
        if not args.resume or not last_path.is_file():
            if not args.init_prototype_checkpoint or not args.init_prototype_checkpoint.is_file():
                raise FileNotFoundError('ConvNeXt robust training requires --init-prototype-checkpoint')
            ck_origin=_read(args.init_prototype_checkpoint)
        cfg=dict(ck_origin.get('prototype_config',{})) if ck_origin else (
            dict(_read(last_path).get('prototype_config',{})) if last_path.is_file() else {})
        cfg.pop('pretrained',None)
        model=HybridPrototypePAR(pretrained=False,**cfg)
        if ck_origin is not None:
            state=next((ck_origin[k] for k in ('inference_model_state','ema_model_state','model_state')
                        if k in ck_origin and ck_origin[k] is not None),None)
            if state is None:raise RuntimeError('Original Prototype has no model weights')
            model.load_state_dict(state,strict=True)
            print('[INIT] Original Prototype copied as trainable Robust ConvNeXt student; protected file unchanged',flush=True)
            del ck_origin
    optimizer=None
    start_epoch=0;best_score=-1e9
    settings={'seed':args.seed,'encoder':args.encoder,'heldout_domain':args.heldout_domain,'epochs':args.epochs,
        'steps_per_epoch':args.steps_per_epoch,'batch_size':args.batch_size,'trainable_last':args.trainable_last,
        'asl_weight':args.asl_weight,'supcon_weight':args.supcon_weight,
        'alignment_weight':args.alignment_weight,'consistency_weight':args.consistency_weight,
        'backbone_lr':args.backbone_lr,'head_lr':args.head_lr,'weight_decay':args.weight_decay,
        'sampler_alpha':args.sampler_alpha,'warmup_epochs':args.warmup_epochs,
        'val_limit':args.val_limit}
    if args.resume and last_path.is_file():
        ck=_read(last_path)
        if ck.get('training_settings')!=settings:raise ValueError('Resume settings mismatch; use separate directory')
        model.load_state_dict(ck['model_state'],strict=True)
        start_epoch=int(ck['epoch'])+1
        best_score=float(ck['best_score'])
        print('[RESUME] epoch',start_epoch+1,'of',args.epochs,flush=True)
    elif args.encoder=='dinov2':
        model.load_dino_pretrained()
    model=model.to(device)
    backbone_prefix=('dino.' if args.encoder=='dinov2' else 'backbone.')
    param_head=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(backbone_prefix)]
    param_backbone=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(backbone_prefix)]
    optimizer=torch.optim.AdamW([{'params':param_head,'lr':args.head_lr},
                                 {'params':param_backbone,'lr':args.backbone_lr}],weight_decay=args.weight_decay)
    if args.resume and last_path.is_file():
        optimizer.load_state_dict(ck['optimizer_state'])
        del ck
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    for epoch in tqdm(range(start_epoch,args.epochs),desc='DINOv2 training epochs',unit='epoch',dynamic_ncols=True):
        model.train()
        train_loader.batch_sampler.set_epoch(epoch)
        epoch_values=[]
        for step,batch in enumerate(tqdm(train_loader,desc=f'DINO seed{args.seed} epoch {epoch+1}/{args.epochs}',unit='batch',dynamic_ncols=True)):
            x=batch['image'].to(device,non_blocking=True)
            y=batch['target'].to(device,non_blocking=True)
            v=batch['valid'].to(device,non_blocking=True)
            dom=batch['domain'].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                aux=model(x,return_aux=True)
                main=masked_asl(aux['logits'],y,v,pos)
                sep=prototype_separation(aux)
                warm=min(1.,max(0.,(epoch+1-args.warmup_epochs)/max(1,args.warmup_epochs)))
                if warm>0 and args.supcon_weight>0:
                    contrast=cross_domain_supcon(aux['features'],y,v,dom,max_attributes=12)
                else:contrast=main.detach()*0.
                if warm>0 and args.alignment_weight>0:
                    align=domain_positive_negative_alignment(aux['features'],y,v,dom,max_attributes=12)
                else:align=main.detach()*0.
                if warm>0 and args.consistency_weight>0 and step%2==0:
                    aug=mild_domain_augmentation(x)
                    logits2=model(aug)
                    consistent=(torch.sigmoid(logits2.float())-torch.sigmoid(aux['logits'].float()).detach()).square().mean()
                else:consistent=main.detach()*0.
                loss=(args.asl_weight*main+args.separation_weight*sep+
                      warm*(args.supcon_weight*contrast+args.alignment_weight*align+
                            args.consistency_weight*consistent))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DINO loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            if args.encoder=='convnext':
                model.update_prototype_ema(aux['features'],y,v)
            epoch_values.append(float(loss.item()))
            if (step+1)%100==0:
                print(f'[TRAIN] epoch={epoch+1}/{args.epochs} step={step+1}/{args.steps_per_epoch} loss={np.mean(epoch_values[-100:]):.4f} ASL={main.item():.4f} SupCon={contrast.item():.4f} Align={align.item():.4f} Cons={consistent.item():.4f}',flush=True)
        score,metrics,domain_scores=evaluate(model,val_loader,device)
        print(f'[VALID] epoch={epoch+1} loss={np.mean(epoch_values):.4f} robust={score:.6f} mA={metrics["mA"]:.6f} F1={metrics["instance_f1"]:.6f}',flush=True)
        if score>best_score:
            best_score=float(score)
            model_config=({'dinov2_config':config_for_checkpoint(model)} if args.encoder=='dinov2'
                          else {'prototype_config':cfg})
            torch.save({'epoch':epoch,'selection':best_score,'inference_model_state':{k:t.detach().cpu() for k,t in model.state_dict().items()},
                        **model_config,'validation_metrics':metrics,'validation_by_domain':domain_scores,
                        'pretrained_backbone':('facebook/dinov2-small' if args.encoder=='dinov2' else 'protected_prototype'),
                        'training_settings':settings},best_path)
            print('[BEST] Saved new DINO checkpoint:',best_path,flush=True)
        # Always save resume checkpoint AFTER each fully completed epoch.
        torch.save({'epoch':epoch,'best_score':best_score,'model_state':{k:t.detach().cpu() for k,t in model.state_dict().items()},
                    'prototype_config':(cfg if args.encoder=='convnext' else {}),
                    'optimizer_state':optimizer.state_dict(),'training_settings':settings},last_path)
    print('[DONE] Best source-FIT checkpoint:',best_path,flush=True)
    if args.heldout_domain>=0:
        # Held-out domain is evaluated once after training/checkpoint selection.
        hold=np.flatnonzero(audit & (val_ds.domains==args.heldout_domain))
        if len(hold):
            model.load_state_dict(_read(best_path)['inference_model_state'])
            hold_loader=DataLoader(Subset(val_ds,hold.tolist()),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
            _,met,bydom=evaluate(model,hold_loader,device)
            report=best_path.with_suffix('.json')
            report.write_text(json.dumps({'heldout':args.heldout_domain,'metrics':met,'by_domain':bydom},indent=2))
            print('[LODO HELDOUT]',met,flush=True)

if __name__=='__main__':main()
