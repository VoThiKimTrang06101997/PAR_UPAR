import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import ATTRIBUTE_NAMES
from .prompts import build_prompt_bank

def l2n(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

class VAPORPAR(nn.Module):
    """
    VAPOR-PAR:
      SigLIP2 + semantic attribute queries + attribute cross-attention
      + positive/negative language/visual dual prototypes
      + prototype-margin classification.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_attributes = cfg.num_attributes
        self.vlm = AutoModel.from_pretrained(
            cfg.model_id,
            cache_dir=(cfg.hf_cache_dir or None),
        )
        for p in self.vlm.parameters():
            p.requires_grad = False

        self.hidden_dim = int(self.vlm.config.vision_config.hidden_size)
        self.text_dim = int(getattr(self.vlm.config.text_config, "hidden_size", self.hidden_dim))

        self.learned_queries = nn.Parameter(torch.randn(self.num_attributes, self.hidden_dim) * 0.02)
        self.semantic_query_proj = nn.Linear(self.text_dim, self.hidden_dim, bias=False)
        self.lang_to_hidden = nn.Linear(self.text_dim, self.hidden_dim, bias=False)
        if self.text_dim == self.hidden_dim:
            nn.init.eye_(self.semantic_query_proj.weight)
            nn.init.eye_(self.lang_to_hidden.weight)
        else:
            nn.init.xavier_uniform_(self.semantic_query_proj.weight)
            nn.init.xavier_uniform_(self.lang_to_hidden.weight)

        self.cross_attn = nn.MultiheadAttention(
            self.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )

        self.visual_pos = nn.Parameter(torch.randn(self.num_attributes, self.hidden_dim) * 0.02)
        self.visual_neg = nn.Parameter(torch.randn(self.num_attributes, self.hidden_dim) * 0.02)
        self.gate_logits = nn.Parameter(torch.zeros(self.num_attributes, 1))
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.temperature), dtype=torch.float32))

        self.register_buffer("lang_pos", torch.empty(0), persistent=True)
        self.register_buffer("lang_neg", torch.empty(0), persistent=True)
        self.register_buffer("lang_pos_variants", torch.empty(0), persistent=True)
        self.register_buffer("lang_neg_variants", torch.empty(0), persistent=True)
        self.prompts_initialized = False

    @torch.no_grad()
    def initialize_prompts(self, processor, device, batch_size=64):
        bank = build_prompt_bank(ATTRIBUTE_NAMES)
        pos_lists = [bank[a]["positive"] for a in ATTRIBUTE_NAMES]
        neg_lists = [bank[a]["negative"] for a in ATTRIBUTE_NAMES]
        k = len(pos_lists[0])

        all_text = []
        for c in range(self.num_attributes):
            all_text.extend(pos_lists[c])
            all_text.extend(neg_lists[c])

        chunks = []
        self.vlm.eval()
        for s in range(0, len(all_text), batch_size):
            toks = processor(
                text=all_text[s:s+batch_size],
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_text_length,
                return_tensors="pt",
            )
            toks = {kk: vv.to(device) for kk, vv in toks.items() if kk in ("input_ids","attention_mask","position_ids")}
            feats = self.vlm.get_text_features(**toks)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            chunks.append(feats.detach().float().cpu())

        feats = torch.cat(chunks, 0).view(self.num_attributes, 2, k, -1)
        posv = l2n(feats[:,0], dim=-1)
        negv = l2n(feats[:,1], dim=-1)
        pos = l2n(posv.mean(1), dim=-1)
        neg = l2n(negv.mean(1), dim=-1)

        self.lang_pos = pos.to(device)
        self.lang_neg = neg.to(device)
        self.lang_pos_variants = posv.to(device)
        self.lang_neg_variants = negv.to(device)
        self.prompts_initialized = True

        # Semantic initialization of visual prototypes.
        pos_h = l2n(self.lang_to_hidden(self.lang_pos), -1)
        neg_h = l2n(self.lang_to_hidden(self.lang_neg), -1)
        self.visual_pos.copy_(pos_h)
        self.visual_neg.copy_(neg_h)

    def unfreeze_last_vision_blocks(self, n=2):
        """Optional Stage-C adaptation without fully fine-tuning SigLIP2."""
        for p in self.vlm.parameters():
            p.requires_grad = False
        candidates = []
        for name in ("encoder", "vision_model"):
            obj = getattr(self.vlm.vision_model, name, None)
            if obj is not None:
                candidates.append(obj)
        blocks = None
        for obj in candidates:
            for attr in ("layers", "layer"):
                x = getattr(obj, attr, None)
                if x is not None:
                    blocks = x
                    break
            if blocks is not None:
                break
        if blocks is None:
            print("[WARN] Could not locate SigLIP2 vision blocks; backbone stays frozen.")
            return 0
        n = min(int(n), len(blocks))
        for block in list(blocks)[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"[Stage C] Unfroze last {n} SigLIP2 vision blocks.")
        return n

    def _vision_forward(self, inputs):
        kwargs = {}
        for k in ("pixel_values","pixel_attention_mask","spatial_shapes"):
            if k in inputs:
                kwargs[k] = inputs[k]
        backbone_trainable = any(p.requires_grad for p in self.vlm.vision_model.parameters())
        if backbone_trainable:
            out = self.vlm.vision_model(**kwargs)
        else:
            with torch.no_grad():
                out = self.vlm.vision_model(**kwargs)
        return out.last_hidden_state

    def language_hidden(self):
        lp = l2n(self.lang_to_hidden(self.lang_pos), -1)
        ln = l2n(self.lang_to_hidden(self.lang_neg), -1)
        lpv = l2n(self.lang_to_hidden(self.lang_pos_variants), -1)
        lnv = l2n(self.lang_to_hidden(self.lang_neg_variants), -1)
        return lp, ln, lpv, lnv

    def forward(self, inputs):
        if not self.prompts_initialized and self.lang_pos.numel() == 0:
            raise RuntimeError("Call model.initialize_prompts(processor, device) before forward().")

        patches = self._vision_forward(inputs)                    # [B, M, D]
        semantic_dir = l2n(self.lang_pos - self.lang_neg, -1)   # [C, Dt]
        queries = self.learned_queries + self.semantic_query_proj(semantic_dir)
        queries = queries.unsqueeze(0).expand(patches.size(0), -1, -1)

        key_padding_mask = None
        pam = inputs.get("pixel_attention_mask", None)
        if pam is not None and pam.ndim == 2 and pam.shape[1] == patches.shape[1]:
            key_padding_mask = ~pam.bool()

        attended, attn = self.cross_attn(
            self.query_norm(queries), patches, patches,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = queries + attended
        h = h + self.ffn(h)
        h = l2n(h, -1)

        lang_pos_h, lang_neg_h, pos_var_h, neg_var_h = self.language_hidden()
        gate = torch.sigmoid(self.gate_logits)
        r_pos = l2n(gate * self.visual_pos + (1.0-gate) * lang_pos_h, -1)
        r_neg = l2n(gate * self.visual_neg + (1.0-gate) * lang_neg_h, -1)

        s_pos = (h * r_pos.unsqueeze(0)).sum(-1)
        s_neg = (h * r_neg.unsqueeze(0)).sum(-1)
        tau = self.log_tau.exp().clamp(0.01, 1.0)
        logits = (s_pos - s_neg) / tau

        return {
            "logits": logits,
            "features": h,
            "s_pos": s_pos,
            "s_neg": s_neg,
            "lang_pos_h": lang_pos_h,
            "lang_neg_h": lang_neg_h,
            "pos_var_h": pos_var_h,
            "neg_var_h": neg_var_h,
            "fused_pos": r_pos,
            "fused_neg": r_neg,
        }
