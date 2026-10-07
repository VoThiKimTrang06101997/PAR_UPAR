from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _threshold_vector(thresholds, n_attributes: int) -> np.ndarray:
    th = np.asarray(thresholds, dtype=np.float32)
    if th.ndim == 0:
        th = np.full(n_attributes, float(th), dtype=np.float32)
    th = th.reshape(-1)
    if th.size != n_attributes:
        raise ValueError(f"Expected {n_attributes} thresholds, got {th.size}")
    return th


def mean_accuracy(y_true, y_pred) -> float:
    y_true = _np(y_true).astype(np.int32)
    y_pred = _np(y_pred).astype(np.int32)
    vals = []
    for c in range(y_true.shape[1]):
        valid = (y_true[:, c] == 0) | (y_true[:, c] == 1)
        yt = y_true[valid, c]
        yp = y_pred[valid, c]
        if yt.size == 0:
            continue
        pos = yt == 1
        neg = yt == 0
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        tpr = ((yp == 1) & pos).sum() / max(int(pos.sum()), 1)
        tnr = ((yp == 0) & neg).sum() / max(int(neg.sum()), 1)
        vals.append(0.5 * (tpr + tnr))
    return float(np.mean(vals)) if vals else 0.0


def label_f1(y_true, y_pred) -> float:
    y_true = _np(y_true).astype(np.int32)
    y_pred = _np(y_pred).astype(np.int32)
    vals = []
    for c in range(y_true.shape[1]):
        valid = (y_true[:, c] == 0) | (y_true[:, c] == 1)
        yt = y_true[valid, c]
        yp = y_pred[valid, c]
        if yt.size == 0:
            continue
        tp = np.logical_and(yt == 1, yp == 1).sum()
        fp = np.logical_and(yt == 0, yp == 1).sum()
        fn = np.logical_and(yt == 1, yp == 0).sum()
        den = 2 * tp + fp + fn
        vals.append(float(2 * tp / den) if den > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def instance_metrics(y_true, y_pred):
    y_true = _np(y_true).astype(np.int32)
    y_pred = _np(y_pred).astype(np.int32)
    valid = (y_true == 0) | (y_true == 1)
    yt = (y_true == 1) & valid
    yp = (y_pred == 1) & valid
    tp = np.logical_and(yt, yp).sum(axis=1).astype(np.float64)
    fp = np.logical_and(~yt, yp).sum(axis=1).astype(np.float64)
    fn = np.logical_and(yt, ~yp).sum(axis=1).astype(np.float64)
    union = tp + fp + fn
    pred_n = tp + fp
    true_n = tp + fn
    acc = np.divide(tp, union, out=np.ones_like(tp), where=union > 0)
    prec = np.divide(tp, pred_n, out=np.ones_like(tp), where=pred_n > 0)
    rec = np.divide(tp, true_n, out=np.ones_like(tp), where=true_n > 0)
    den = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, den, out=np.ones_like(tp), where=den > 0)
    return float(acc.mean()), float(prec.mean()), float(rec.mean()), float(f1.mean())


def harmonic_challenge_average(ma: float, instance_f1: float) -> float:
    ma = float(ma)
    instance_f1 = float(instance_f1)
    return float(2.0 * ma * instance_f1 / max(ma + instance_f1, EPS))


def evaluate_probs(y_true, probs, thresholds=0.5):
    y_true = _np(y_true).astype(np.int32)
    probs = _np(probs).astype(np.float32)
    th = _threshold_vector(thresholds, probs.shape[1])
    pred = (probs >= th.reshape(1, -1)).astype(np.int32)
    ma = mean_accuracy(y_true, pred)
    lf1 = label_f1(y_true, pred)
    iacc, iprec, irec, if1 = instance_metrics(y_true, pred)
    score = harmonic_challenge_average(ma, if1)
    return {
        "mA": ma,
        "Label_F1": lf1,
        "Instance_Acc": iacc,
        "instance_precision": iprec,
        "instance_recall": irec,
        "instance_f1": if1,
        "Challenge_Avg": score,
        "challenge_score": score,
    }


def robust_selection_score(y_true, probs, thresholds=0.5, domains=None):
    overall = evaluate_probs(y_true, probs, thresholds)
    if domains is None:
        return overall["Challenge_Avg"], overall, {}
    domains = np.asarray(domains, dtype=np.int64)
    per_domain = {}
    vals = []
    for d in sorted(set(int(x) for x in domains.tolist() if int(x) >= 0)):
        mask = domains == d
        if not mask.any():
            continue
        m = evaluate_probs(_np(y_true)[mask], _np(probs)[mask], thresholds)
        per_domain[d] = m
        vals.append(m["Challenge_Avg"])
    if not vals:
        return overall["Challenge_Avg"], overall, per_domain
    # Hidden-domain selection: keep overall quality but protect the weakest domain.
    robust = (
        0.45 * overall["Challenge_Avg"]
        + 0.30 * float(np.mean(vals))
        + 0.25 * float(np.min(vals))
    )
    return float(robust), overall, per_domain


def _attribute_ma_thresholds(y_true, probs, grid=None):
    """Per-attribute thresholds that maximize balanced accuracy / mA contribution."""
    y_true = _np(y_true).astype(np.int32)
    probs = _np(probs).astype(np.float32)
    if grid is None:
        grid = np.arange(0.18, 0.821, 0.02, dtype=np.float32)
    out = np.full(probs.shape[1], 0.5, dtype=np.float32)
    for c in range(probs.shape[1]):
        valid = (y_true[:, c] == 0) | (y_true[:, c] == 1)
        yt = y_true[valid, c]
        pp = probs[valid, c]
        if yt.size == 0:
            continue
        pos = yt == 1
        neg = yt == 0
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        best_s, best_t = -1.0, 0.5
        for t in grid:
            yp = pp >= float(t)
            tpr = float(yp[pos].mean())
            tnr = float((~yp[neg]).mean())
            s = 0.5 * (tpr + tnr)
            if s > best_s:
                best_s, best_t = s, float(t)
        out[c] = best_t
    return out


def tune_lodo_thresholds(y_true, probs, domains, grid=None):
    """
    Leave-one-domain-out threshold calibration for hidden-domain PAR.

    The per-fold attribute thresholds are precomputed once, so this remains
    practical on the 33k-image validation split.  The objective penalizes
    recall-heavy operating points because the supplied hidden row had
    precision far below recall (0.5633 vs 0.8022).
    """
    y_true = _np(y_true).astype(np.int32)
    probs = _np(probs).astype(np.float32)
    domains = np.asarray(domains, dtype=np.int64)
    ids = sorted(set(int(x) for x in domains.tolist() if int(x) >= 0))
    if len(ids) < 2:
        attr = _attribute_ma_thresholds(y_true, probs, grid)
        return attr, {"mode": "attr_fallback"}

    fold_cache = {}
    for d in ids:
        fit = domains != d
        hold = domains == d
        if fit.any() and hold.any():
            fold_cache[d] = {
                "hold": hold,
                "attr": _attribute_ma_thresholds(y_true[fit], probs[fit], grid),
            }

    candidates = [
        (float(global_t), float(alpha), float(shift))
        for global_t in np.arange(0.46, 0.661, 0.025, dtype=np.float32)
        for alpha in (0.20, 0.35, 0.50, 0.65)
        for shift in (0.00, 0.015, 0.030, 0.045, 0.060)
    ]

    best_obj = -1e9
    best_params = None
    best_folds = None

    for global_t, alpha, shift in candidates:
        scores, weak_scores, recall_gaps = [], [], []
        folds = {}
        for d, cache in fold_cache.items():
            th = (1.0 - alpha) * global_t + alpha * cache["attr"] + shift
            th = np.clip(th, 0.30, 0.84).astype(np.float32)
            hold = cache["hold"]
            m = evaluate_probs(y_true[hold], probs[hold], th)
            scores.append(m["Challenge_Avg"])
            weak_scores.append(min(m["mA"], m["instance_f1"]))
            recall_gaps.append(max(0.0, m["instance_recall"] - m["instance_precision"]))
            folds[int(d)] = m
        if not scores:
            continue
        mean_s = float(np.mean(scores))
        min_s = float(np.min(scores))
        min_component = float(np.mean(weak_scores))
        std_s = float(np.std(scores))
        recall_gap = float(np.mean(recall_gaps))
        obj = (
            0.50 * mean_s
            + 0.25 * min_s
            + 0.25 * min_component
            - 0.05 * std_s
            - 0.07 * recall_gap
        )
        if obj > best_obj:
            best_obj = obj
            best_params = (global_t, alpha, shift)
            best_folds = folds

    if best_params is None:
        raise RuntimeError("LODO calibration found no valid candidate")

    global_t, alpha, shift = best_params
    attr_all = _attribute_ma_thresholds(y_true, probs, grid)
    thresholds = (1.0 - alpha) * global_t + alpha * attr_all + shift
    thresholds = np.clip(thresholds, 0.30, 0.84).astype(np.float32)
    final = evaluate_probs(y_true, probs, thresholds)
    return thresholds, {
        "mode": "lodo_precision_aware",
        "objective": float(best_obj),
        "global_t": float(global_t),
        "alpha": float(alpha),
        "shift": float(shift),
        "folds": best_folds,
        "final_mixed": final,
    }


# Backward-compatible alias used by older notebooks.
def tune_attribute_thresholds(y_true, probs, grid=None, domains=None):
    if domains is not None:
        return tune_lodo_thresholds(y_true, probs, domains, grid=grid)[0]
    return _attribute_ma_thresholds(y_true, probs, grid=grid)


