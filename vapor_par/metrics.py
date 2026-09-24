import numpy as np
import torch

EPS = 1e-12

def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def mean_accuracy(y_true, y_pred):
    y_true, y_pred = _np(y_true).astype(np.int32), _np(y_pred).astype(np.int32)
    vals = []
    for c in range(y_true.shape[1]):
        pos = y_true[:,c] == 1
        neg = ~pos
        tpr = ((y_pred[:,c] == 1) & pos).sum() / max(pos.sum(), 1)
        tnr = ((y_pred[:,c] == 0) & neg).sum() / max(neg.sum(), 1)
        vals.append(0.5*(tpr+tnr))
    return float(np.mean(vals))

def instance_metrics(y_true, y_pred):
    y_true, y_pred = _np(y_true).astype(bool), _np(y_pred).astype(bool)
    inter = (y_true & y_pred).sum(1).astype(np.float64)
    pred_n = y_pred.sum(1).astype(np.float64)
    true_n = y_true.sum(1).astype(np.float64)
    precision_i = np.divide(inter, pred_n, out=np.ones_like(inter), where=pred_n>0)
    recall_i = np.divide(inter, true_n, out=np.ones_like(inter), where=true_n>0)
    p = float(precision_i.mean())
    r = float(recall_i.mean())
    f1 = 2*p*r/max(p+r, EPS)
    return p, r, f1

def evaluate_probs(y_true, probs, thresholds=0.5):
    y_true, probs = _np(y_true), _np(probs)
    th = np.asarray(thresholds, dtype=np.float32)
    if th.ndim == 0:
        th = np.full(probs.shape[1], float(th), dtype=np.float32)
    pred = (probs >= th.reshape(1,-1)).astype(np.int32)
    ma = mean_accuracy(y_true, pred)
    p, r, f1 = instance_metrics(y_true, pred)
    return {
        "mA": ma, "instance_precision": p, "instance_recall": r,
        "instance_f1": f1, "challenge_score": 0.5*(ma+f1)
    }

def tune_attribute_thresholds(y_true, probs, grid=None):
    """Robust simple per-attribute balanced-accuracy threshold search."""
    y_true, probs = _np(y_true), _np(probs)
    if grid is None:
        grid = np.arange(0.10, 0.901, 0.025)
    C = probs.shape[1]
    out = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        best_s, best_t = -1, 0.5
        yt = y_true[:,c].astype(np.int32)
        for t in grid:
            yp = (probs[:,c] >= t).astype(np.int32)
            pos = yt == 1
            neg = yt == 0
            tpr = ((yp==1)&pos).sum()/max(pos.sum(),1)
            tnr = ((yp==0)&neg).sum()/max(neg.sum(),1)
            s = 0.5*(tpr+tnr)
            if s > best_s:
                best_s, best_t = s, float(t)
        out[c] = best_t
    return out
