"""Standalone PyTorch DINOv2-S/14 + 40 attribute queries + hybrid prototypes.

The DINOv2 pretrained tensor mapping is performed only in Colab using transformers.
Inference depends only on PyTorch, torchvision and PIL (no network/HF import).
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class DinoBlock(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        a = self.norm1(x)
        batch, length, dim = a.shape
        q = self.q(a).reshape(batch,length,self.heads,-1).transpose(1,2)
        k = self.k(a).reshape(batch,length,self.heads,-1).transpose(1,2)
        v = self.v(a).reshape(batch,length,self.heads,-1).transpose(1,2)
        a = F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        a = a.transpose(1,2).reshape(batch,length,dim)
        x = x + self.ls1 * self.attn_out(a)
        a = self.norm2(x)
        return x + self.ls2 * self.fc2(F.gelu(self.fc1(a)))

class DinoSmall(nn.Module):
    def __init__(self, layers=12, dim=384, heads=6, patch=14):
        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Conv2d(3,dim,kernel_size=patch,stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1370,dim))
        self.blocks = nn.ModuleList([DinoBlock(dim,heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim,eps=1e-6)
        self.trainable_last = 0

    def set_trainable_last(self, count):
        count=int(count)
        if not 0 <= count <= len(self.blocks):
            raise ValueError('Invalid number of trainable DINO blocks')
        self.requires_grad_(False)
        for block in list(self.blocks)[len(self.blocks)-count:]:
            block.requires_grad_(True)
        if count:
            self.norm.requires_grad_(True)
        self.trainable_last=count

    def forward(self,x):
        patch=self.patch_embed(x)
        b,d,h,w=patch.shape
        tokens=patch.flatten(2).transpose(1,2)
        p=self.pos_embed[:,1:]
        side=math.isqrt(p.shape[1])
        if side*side != p.shape[1]:
            raise ValueError('Pretrained DINO position embedding must be square')
        p=p.reshape(1,side,side,d).permute(0,3,1,2)
        p=F.interpolate(p.float(),size=(h,w),mode='bicubic',align_corners=False)
        p=p.to(tokens.dtype).flatten(2).transpose(1,2)
        cls=self.cls_token.expand(b,-1,-1)
        tokens=torch.cat((cls,tokens),dim=1)
        tokens=tokens+torch.cat((self.pos_embed[:,:1],p),dim=1)
        frozen=len(self.blocks)-self.trainable_last
        if self.training and self.trainable_last < len(self.blocks):
            with torch.no_grad():
                for block in list(self.blocks)[:frozen]:
                    tokens=block(tokens)
            tokens=tokens.detach()
            for block in list(self.blocks)[frozen:]:
                tokens=block(tokens)
        else:
            for block in self.blocks:
                tokens=block(tokens)
        return self.norm(tokens)

class DinoAttributePAR(nn.Module):
    def __init__(self, num_attributes=40, prototype_dim=192,
                 prototype_temperature=.22, prototype_gate_init=.25,
                 pretrained=False, trainable_last=0):
        super().__init__()
        self.num_attributes=int(num_attributes)
        self.prototype_dim=int(prototype_dim)
        self.prototype_temperature=float(prototype_temperature)
        self.dino=DinoSmall()
        self.dino.set_trainable_last(trainable_last)
        self.linear_head=nn.Sequential(nn.Dropout(.18),nn.Linear(384,self.num_attributes))
        self.token_norm=nn.LayerNorm(384)
        self.token_proj=nn.Linear(384,self.prototype_dim,bias=False)
        self.queries=nn.Parameter(torch.randn(self.num_attributes,self.prototype_dim)*.02)
        self.attr_norm=nn.LayerNorm(self.prototype_dim)
        self.prototype_pos=nn.Parameter(torch.randn(self.num_attributes,self.prototype_dim)*.03)
        self.prototype_neg=nn.Parameter(torch.randn(self.num_attributes,self.prototype_dim)*.03)
        self.prototype_gate_logits=nn.Parameter(torch.full((self.num_attributes,),math.log(prototype_gate_init/(1.-prototype_gate_init))))
        if pretrained:
            self.load_dino_pretrained()

    @torch.no_grad()
    def load_dino_pretrained(self, model_id='facebook/dinov2-small'):
        """Copy HF pretrained DINOv2-S weights using version-aware, strict tensor mapping.

        Works with legacy `attention.attention.query/key/value` and modern
        `attention.q_proj/k_proj/v_proj` naming. Only used at training init;
        the exported PyTorch runtime never imports Transformers.
        """
        try:
            from transformers import Dinov2Model
        except ImportError as exc:
            raise RuntimeError('Install transformers to initialize pretrained DINOv2') from exc

        print('[DINO] Loading pretrained backbone:', model_id, flush=True)
        hf = Dinov2Model.from_pretrained(model_id).eval()
        src = hf.state_dict()
        own = self.dino.state_dict()
        mapping = {}

        def choose(*candidates):
            """Return exactly one existing HF key, optionally prefixed with dinov2."""
            for candidate in candidates:
                for prefix in ('', 'dinov2.'):
                    key = prefix + candidate
                    if key in src:
                        return key
            raise RuntimeError(
                'HF DINOv2 missing parameter; tried: ' + str(candidates) +
                '; sample attention keys: ' + str([
                    k for k in src if 'encoder.layer.0.attention' in k
                ][:18])
            )

        mapping.update({
            'patch_embed.weight': choose('embeddings.patch_embeddings.projection.weight'),
            'patch_embed.bias': choose('embeddings.patch_embeddings.projection.bias'),
            'cls_token': choose('embeddings.cls_token'),
            'pos_embed': choose('embeddings.position_embeddings'),
            'norm.weight': choose('layernorm.weight'),
            'norm.bias': choose('layernorm.bias'),
        })
        for i in range(len(self.dino.blocks)):
            a = f'blocks.{i}.'
            b = f'encoder.layer.{i}.'
            mapping.update({
                a+'norm1.weight': choose(b+'norm1.weight'),
                a+'norm1.bias': choose(b+'norm1.bias'),
                a+'q.weight': choose(b+'attention.q_proj.weight', b+'attention.attention.query.weight'),
                a+'q.bias': choose(b+'attention.q_proj.bias', b+'attention.attention.query.bias'),
                a+'k.weight': choose(b+'attention.k_proj.weight', b+'attention.attention.key.weight'),
                a+'k.bias': choose(b+'attention.k_proj.bias', b+'attention.attention.key.bias'),
                a+'v.weight': choose(b+'attention.v_proj.weight', b+'attention.attention.value.weight'),
                a+'v.bias': choose(b+'attention.v_proj.bias', b+'attention.attention.value.bias'),
                a+'attn_out.weight': choose(b+'attention.o_proj.weight', b+'attention.output.dense.weight'),
                a+'attn_out.bias': choose(b+'attention.o_proj.bias', b+'attention.output.dense.bias'),
                a+'ls1': choose(b+'layer_scale1.lambda1'),
                a+'ls2': choose(b+'layer_scale2.lambda1'),
                a+'norm2.weight': choose(b+'norm2.weight'),
                a+'norm2.bias': choose(b+'norm2.bias'),
                a+'fc1.weight': choose(b+'mlp.fc1.weight'),
                a+'fc1.bias': choose(b+'mlp.fc1.bias'),
                a+'fc2.weight': choose(b+'mlp.fc2.weight'),
                a+'fc2.bias': choose(b+'mlp.fc2.bias'),
            })
        if set(own) != set(mapping):
            raise RuntimeError('Incomplete DINO mapping; unmapped=' + str(sorted(set(own) - set(mapping))[:16]))
        for dst, key in mapping.items():
            if tuple(own[dst].shape) != tuple(src[key].shape):
                raise ValueError(
                    f'DINO shape mismatch: {dst} <- {key}: '
                    f'{tuple(own[dst].shape)} != {tuple(src[key].shape)}'
                )
            own[dst] = src[key].detach().to(device='cpu', dtype=own[dst].dtype).clone()
        self.dino.load_state_dict(own, strict=True)
        del hf, src
        print(f'[DINO] PRETRAINED LOADED: {len(mapping)} tensors, '
              f'{len(self.dino.blocks)} transformer blocks; checkpoint={model_id}', flush=True)

    def forward(self,x,return_aux=False):
        tokens=self.dino(x)
        linear=self.linear_head(tokens[:,0])
        patch=F.normalize(self.token_proj(self.token_norm(tokens[:,1:])),dim=-1)
        query=F.normalize(self.queries,dim=-1)
        attn=torch.softmax(torch.einsum('bnd,cd->bcn',patch,query)/.12,dim=-1)
        feat=self.attr_norm(torch.einsum('bcn,bnd->bcd',attn,patch))
        feat=F.normalize(feat,dim=-1)
        pos=F.normalize(self.prototype_pos,dim=-1)
        neg=F.normalize(self.prototype_neg,dim=-1)
        proto=(torch.einsum('bcd,cd->bc',feat,pos)-torch.einsum('bcd,cd->bc',feat,neg))/self.prototype_temperature
        gate=torch.sigmoid(self.prototype_gate_logits).clamp(.05,.65).view(1,-1)
        logits=(1-gate)*linear+gate*proto
        if not return_aux:
            return logits
        return {'logits':logits,'linear_logits':linear,'proto_logits':proto,
                'features':feat,'prototype_pos':pos,'prototype_neg':neg,'prototype_gate':gate}


def config_for_checkpoint(model):
    return {'num_attributes':model.num_attributes,'prototype_dim':model.prototype_dim,
            'prototype_temperature':model.prototype_temperature,'trainable_last':0}
