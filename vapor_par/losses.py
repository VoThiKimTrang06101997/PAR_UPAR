import torch
import torch.nn.functional as F
from .config import ONTOLOGY_GROUPS

EPS = 1e-7

def balanced_asymmetric_loss(logits, targets, pos_prior, gamma_pos=1.0, gamma_neg=2.0):
    p = torch.sigmoid(logits).clamp(EPS, 1-EPS)
    prior = pos_prior.to(logits.device).clamp(1e-3, 1-1e-3)
    w_pos = (0.5 / prior).clamp(max=20.0)
    w_neg = (0.5 / (1-prior)).clamp(max=20.0)
    pos = -w_pos * targets * ((1-p) ** gamma_pos) * torch.log(p)
    neg = -w_neg * (1-targets) * (p ** gamma_neg) * torch.log(1-p)
    return (pos + neg).mean()

def semantic_contrastive_loss(features, targets, lang_pos_h, lang_neg_h, tau=0.07):
    # AMP-safe equivalent: softmax([sn, sp])_positive == sigmoid(sp - sn)
    sp = torch.einsum("bcd,cd->bc", features, lang_pos_h) / tau
    sn = torch.einsum("bcd,cd->bc", features, lang_neg_h) / tau
    sem_logits = sp - sn
    return F.binary_cross_entropy_with_logits(sem_logits, targets.float())

def domain_alignment_loss(features, targets, domains):
    """Align positive and negative per-domain class centroids across known source domains."""
    losses = []
    valid_domains = [int(d) for d in domains.unique().tolist() if int(d) >= 0]
    if len(valid_domains) < 2:
        return features.new_zeros(())
    C = targets.shape[1]
    for c in range(C):
        for positive in (1, 0):
            cents = []
            for d in valid_domains:
                mask = (domains == d) & ((targets[:,c] > 0.5) if positive else (targets[:,c] <= 0.5))
                if mask.sum() > 0:
                    cents.append(F.normalize(features[mask,c].mean(0), dim=-1))
            if len(cents) >= 2:
                stack = torch.stack(cents)
                center = F.normalize(stack.mean(0), dim=-1)
                losses.append(((stack-center.unsqueeze(0))**2).mean())
    return torch.stack(losses).mean() if losses else features.new_zeros(())

def semantic_anchor_loss(visual_pos, visual_neg, lang_pos_h, lang_neg_h):
    vp = F.normalize(visual_pos, dim=-1)
    vn = F.normalize(visual_neg, dim=-1)
    lp = F.normalize(lang_pos_h, dim=-1)
    ln = F.normalize(lang_neg_h, dim=-1)
    return (1-(vp*lp).sum(-1)).mean() + (1-(vn*ln).sum(-1)).mean()

def prompt_invariance_loss(features, pos_var_h, neg_var_h):
    # features [B,C,D], variants [C,K,D]
    pos = torch.einsum("bcd,ckd->bck", features, pos_var_h)
    neg = torch.einsum("bcd,ckd->bck", features, neg_var_h)
    margin = pos - neg
    return margin.var(dim=-1, unbiased=False).mean()

def relation_geometry_loss(visual_pos, lang_pos_h):
    vp = F.normalize(visual_pos, dim=-1)
    lp = F.normalize(lang_pos_h.detach(), dim=-1)
    s_vis = vp @ vp.t()
    s_txt = lp @ lp.t()
    return F.mse_loss(s_vis, s_txt)

def ontology_loss(logits, targets):
    losses = []
    for _, ids in ONTOLOGY_GROUPS.items():
        ids = torch.tensor(ids, device=logits.device, dtype=torch.long)
        yg = targets.index_select(1, ids)
        mask = (yg.sum(1) == 1)
        if mask.any():
            lg = logits.index_select(1, ids)[mask]
            tg = yg[mask].argmax(1)
            losses.append(F.cross_entropy(lg, tg))
    return torch.stack(losses).mean() if losses else logits.new_zeros(())

def prediction_consistency_loss(weak_logits, strong_logits, confidence=0.80):
    # AMP-safe: weak probabilities are detached targets; strong branch stays in logit space.
    pw = torch.sigmoid(weak_logits).detach()
    mask = (pw >= confidence) | (pw <= (1-confidence))
    if not mask.any():
        return weak_logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(
        strong_logits[mask],
        pw[mask].float(),
    )

def feature_consistency_loss(weak_features, strong_features):
    return (1 - F.cosine_similarity(weak_features.detach(), strong_features, dim=-1)).mean()

def total_vapor_loss(cfg, weak_out, strong_out, targets, domains, pos_prior, epoch):
    l_bal = balanced_asymmetric_loss(
        weak_out["logits"], targets, pos_prior, cfg.gamma_pos, cfg.gamma_neg
    )
    l_sem = semantic_contrastive_loss(
        weak_out["features"], targets, weak_out["lang_pos_h"], weak_out["lang_neg_h"]
    )
    l_anchor = semantic_anchor_loss(
        weak_out["fused_pos"], weak_out["fused_neg"],
        weak_out["lang_pos_h"], weak_out["lang_neg_h"]
    )
    l_prompt = prompt_invariance_loss(
        weak_out["features"], weak_out["pos_var_h"], weak_out["neg_var_h"]
    )
    l_rel = relation_geometry_loss(weak_out["fused_pos"], weak_out["lang_pos_h"])
    l_onto = ontology_loss(weak_out["logits"], targets)

    if epoch >= cfg.semantic_warmup_epochs:
        l_dg = domain_alignment_loss(weak_out["features"], targets, domains)
    else:
        l_dg = weak_out["logits"].new_zeros(())

    if epoch >= cfg.consistency_start_epoch and strong_out is not None:
        l_cons = prediction_consistency_loss(weak_out["logits"], strong_out["logits"])
        l_feat = feature_consistency_loss(weak_out["features"], strong_out["features"])
    else:
        l_cons = weak_out["logits"].new_zeros(())
        l_feat = weak_out["logits"].new_zeros(())

    total = (
        cfg.w_bal*l_bal + cfg.w_sem*l_sem + cfg.w_dg*l_dg +
        cfg.w_anchor*l_anchor + cfg.w_prompt*l_prompt + cfg.w_rel*l_rel +
        cfg.w_onto*l_onto + cfg.w_cons*l_cons + cfg.w_featcons*l_feat
    )
    parts = {
        "loss": total, "bal": l_bal, "sem": l_sem, "dg": l_dg,
        "anchor": l_anchor, "prompt": l_prompt, "rel": l_rel,
        "onto": l_onto, "cons": l_cons, "featcons": l_feat,
    }
    return total, parts
