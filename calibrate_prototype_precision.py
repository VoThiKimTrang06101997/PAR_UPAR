from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from strong_par import (
    ATTRIBUTE_NAMES,
    DOMAIN_NAMES,
    ImageResolver,
    UPARDataset,
    build_eval_transform,
    seed_everything,
)
from prototype_par import HybridPrototypePAR
from vapor_par.metrics import evaluate_probs, robust_selection_score


ATTRIBUTE_GROUPS = {
    "age": list(range(0, 3)),
    "gender": [3],
    "hair": list(range(4, 7)),
    "upper_body": list(range(7, 20)),
    "lower_body": list(range(20, 33)),
    "lower_type": list(range(33, 35)),
    "accessories": list(range(35, 40)),
}

GROUP_NAMES = list(ATTRIBUTE_GROUPS)
GROUP_SCALE_GRID = (0.00, 0.25, 0.50, 0.75, 1.00)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/content/UPAR-Challenge-2027"),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--members",
        type=str,
        default="42,123",
    )
    p.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="prototype_convnext_tiny",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--tta",
        choices=["none", "flip"],
        default="flip",
    )
    p.add_argument(
        "--gate-max",
        type=float,
        default=0.60,
    )
    p.add_argument(
        "--coordinate-passes",
        type=int,
        default=2,
    )
    return p.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def parse_members(spec):
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("No ensemble members")
    return out


def checkpoint_path(root, prefix, seed):
    return root / f"{prefix}_seed{seed}_best.pt"


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def group_scale_vector(group_scales):
    out = np.ones(40, dtype=np.float32)
    for name, indices in ATTRIBUTE_GROUPS.items():
        out[indices] = float(group_scales[name])
    return out


def compose_hybrid_logits(
    linear_logits,
    proto_logits,
    learned_gate,
    group_scales,
    gate_max=0.60,
):
    scale = group_scale_vector(group_scales).reshape(1, -1)
    gate = np.asarray(learned_gate, dtype=np.float32).reshape(1, -1)
    gate = np.clip(
        gate * scale,
        0.0,
        float(gate_max),
    )
    return (
        (1.0 - gate) * linear_logits
        + gate * proto_logits
    ).astype(np.float32)


def build_val_loader(args):
    data_root = args.repo_root / "data"
    val_csv = data_root / "annotations" / "task1" / "val" / "gt.csv"

    if not val_csv.exists():
        raise FileNotFoundError(val_csv)

    df = pd.read_csv(val_csv)
    resolver = ImageResolver(data_root, args.repo_root)

    ds = UPARDataset(
        df,
        resolver,
        build_eval_transform(288, 144),
    )

    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )


@torch.inference_mode()
def predict_components(model, loader, device, use_flip=True):
    model.eval()

    linear_all = []
    proto_all = []
    ys = []
    ds = []

    for batch in tqdm(
        loader,
        desc="Validation components + flip-TTA" if use_flip else "Validation components",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.25,
        file=sys.stdout,
        leave=True,
    ):
        x = batch["image"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            a = model(x, return_aux=True)
            linear = a["linear_logits"]
            proto = a["proto_logits"]

            if use_flip:
                b = model(
                    torch.flip(x, dims=[3]),
                    return_aux=True,
                )
                linear = 0.5 * (
                    linear + b["linear_logits"]
                )
                proto = 0.5 * (
                    proto + b["proto_logits"]
                )

        linear_all.append(
            linear.float().cpu().numpy()
        )
        proto_all.append(
            proto.float().cpu().numpy()
        )

        y = batch["target"].numpy()
        valid = batch["valid"].numpy()
        ys.append(
            np.where(
                valid > 0,
                y,
                -1,
            ).astype(np.int32)
        )
        ds.append(
            np.asarray(
                batch["domain"],
                dtype=np.int64,
            )
        )

    learned_gate = torch.sigmoid(
        model.prototype_gate_logits.detach().float().cpu()
    ).numpy()

    return (
        np.concatenate(linear_all),
        np.concatenate(proto_all),
        learned_gate.astype(np.float32),
        np.concatenate(ys),
        np.concatenate(ds),
    )


def domain_objective(overall, robust, per_domain):
    domain_scores = [
        float(v["Challenge_Avg"])
        for v in per_domain.values()
    ]

    min_domain = (
        min(domain_scores)
        if domain_scores
        else float(overall["Challenge_Avg"])
    )

    precision = float(overall["instance_precision"])
    recall = float(overall["instance_recall"])
    f1 = float(overall["instance_f1"])
    ma = float(overall["mA"])

    recall_gap = max(
        0.0,
        recall - precision,
    )

    imbalance = abs(
        recall - precision
    )

    return float(
        0.30 * float(overall["Challenge_Avg"])
        + 0.25 * float(robust)
        + 0.15 * float(min_domain)
        + 0.15 * min(ma, f1)
        + 0.10 * f1
        + 0.05 * precision
        - 0.22 * recall_gap
        - 0.05 * imbalance
    )


def fast_operating_point(
    y_true,
    logits,
    domains,
):
    probs = sigmoid_np(logits).astype(np.float32)

    best = None

    for t in np.arange(
        0.48,
        0.721,
        0.03,
        dtype=np.float32,
    ):
        robust, overall, per_domain = robust_selection_score(
            y_true,
            probs,
            float(t),
            domains,
        )

        obj = domain_objective(
            overall,
            robust,
            per_domain,
        )

        rec = {
            "objective": float(obj),
            "threshold": float(t),
            "overall": overall,
            "robust": float(robust),
            "per_domain": per_domain,
        }

        if (
            best is None
            or rec["objective"] > best["objective"]
        ):
            best = rec

    return best


def optimize_group_scales(
    y_true,
    linear_logits,
    proto_logits,
    learned_gate,
    domains,
    gate_max,
    coordinate_passes,
):
    # Step 1: global prototype strength.
    best = None

    for scale in GROUP_SCALE_GRID:
        gs = {
            name: float(scale)
            for name in GROUP_NAMES
        }

        logits = compose_hybrid_logits(
            linear_logits,
            proto_logits,
            learned_gate,
            gs,
            gate_max=gate_max,
        )

        rec = fast_operating_point(
            y_true,
            logits,
            domains,
        )
        rec["group_scales"] = dict(gs)

        if (
            best is None
            or rec["objective"] > best["objective"]
        ):
            best = rec

    group_scales = dict(
        best["group_scales"]
    )

    # Step 2: coordinate descent by semantic attribute group.
    for pass_idx in range(
        int(coordinate_passes)
    ):
        print(
            f"  group-scale coordinate pass {pass_idx + 1}/{coordinate_passes}",
            flush=True,
        )

        for group in GROUP_NAMES:
            local_best = None

            for scale in GROUP_SCALE_GRID:
                candidate = dict(
                    group_scales
                )
                candidate[group] = float(scale)

                logits = compose_hybrid_logits(
                    linear_logits,
                    proto_logits,
                    learned_gate,
                    candidate,
                    gate_max=gate_max,
                )

                rec = fast_operating_point(
                    y_true,
                    logits,
                    domains,
                )
                rec["group_scales"] = dict(
                    candidate
                )

                if (
                    local_best is None
                    or rec["objective"]
                    > local_best["objective"]
                ):
                    local_best = rec

            group_scales = dict(
                local_best["group_scales"]
            )

    logits = compose_hybrid_logits(
        linear_logits,
        proto_logits,
        learned_gate,
        group_scales,
        gate_max=gate_max,
    )

    final = fast_operating_point(
        y_true,
        logits,
        domains,
    )
    final["group_scales"] = dict(
        group_scales
    )

    return final, logits


def ensemble_weight_candidates(n):
    if n == 1:
        return [
            np.asarray(
                [1.0],
                dtype=np.float32,
            )
        ]

    if n == 2:
        return [
            np.asarray(
                [w, 1.0 - w],
                dtype=np.float32,
            )
            for w in np.arange(
                0.0,
                1.0001,
                0.10,
            )
        ]

    out = []

    for i in range(n):
        w = np.zeros(
            n,
            dtype=np.float32,
        )
        w[i] = 1.0
        out.append(w)

    out.append(
        np.full(
            n,
            1.0 / n,
            dtype=np.float32,
        )
    )

    return out


def ensemble_logits(logits_list, weights):
    out = np.zeros_like(
        logits_list[0],
        dtype=np.float64,
    )

    for weight, logits in zip(
        weights,
        logits_list,
    ):
        out += (
            float(weight)
            * logits.astype(np.float64)
        )

    return out.astype(np.float32)


def attribute_ma_thresholds(
    y_true,
    probs,
    grid=None,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )
    probs = np.asarray(
        probs,
        dtype=np.float32,
    )

    if grid is None:
        grid = np.arange(
            0.20,
            0.861,
            0.02,
            dtype=np.float32,
        )

    out = np.full(
        probs.shape[1],
        0.5,
        dtype=np.float32,
    )

    for c in range(
        probs.shape[1]
    ):
        valid = (
            (y_true[:, c] == 0)
            | (y_true[:, c] == 1)
        )

        yt = y_true[
            valid,
            c,
        ]
        pp = probs[
            valid,
            c,
        ]

        if yt.size == 0:
            continue

        pos = yt == 1
        neg = yt == 0

        if (
            pos.sum() == 0
            or neg.sum() == 0
        ):
            continue

        best_score = -1.0
        best_t = 0.5

        for t in grid:
            yp = pp >= float(t)
            tpr = float(
                yp[pos].mean()
            )
            tnr = float(
                (~yp[neg]).mean()
            )
            score = 0.5 * (
                tpr + tnr
            )

            if score > best_score:
                best_score = score
                best_t = float(t)

        out[c] = best_t

    return out


def tune_precision_lodo_thresholds(
    y_true,
    probs,
    domains,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )
    probs = np.asarray(
        probs,
        dtype=np.float32,
    )
    domains = np.asarray(
        domains,
        dtype=np.int64,
    )

    ids = sorted(
        set(
            int(x)
            for x in domains.tolist()
            if int(x) >= 0
        )
    )

    if len(ids) < 2:
        thresholds = attribute_ma_thresholds(
            y_true,
            probs,
        )
        return thresholds, {
            "mode": "attribute_fallback",
        }

    fold_cache = {}

    for d in ids:
        fit = domains != d
        hold = domains == d

        if (
            fit.any()
            and hold.any()
        ):
            fold_cache[d] = {
                "hold": hold,
                "attr": attribute_ma_thresholds(
                    y_true[fit],
                    probs[fit],
                ),
            }

    candidates = [
        (
            float(global_t),
            float(alpha),
            float(shift),
        )
        for global_t in np.arange(
            0.48,
            0.741,
            0.02,
            dtype=np.float32,
        )
        for alpha in (
            0.15,
            0.30,
            0.45,
            0.60,
        )
        for shift in (
            0.00,
            0.03,
            0.06,
            0.09,
            0.12,
        )
    ]

    best = None

    for global_t, alpha, shift in tqdm(
        candidates,
        desc="Precision-aware LODO threshold search",
        unit="candidate",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
    ):
        fold_metrics = {}
        fold_scores = []

        for d, cache in fold_cache.items():
            thresholds = (
                (1.0 - alpha)
                * global_t
                + alpha
                * cache["attr"]
                + shift
            )

            thresholds = np.clip(
                thresholds,
                0.34,
                0.88,
            ).astype(np.float32)

            hold = cache["hold"]

            robust, overall, _ = robust_selection_score(
                y_true[hold],
                probs[hold],
                thresholds,
                None,
            )

            # robust == overall score when domains=None.
            obj = domain_objective(
                overall,
                float(robust),
                {},
            )

            fold_metrics[int(d)] = overall
            fold_scores.append(
                float(obj)
            )

        if not fold_scores:
            continue

        mean_obj = float(
            np.mean(fold_scores)
        )
        min_obj = float(
            np.min(fold_scores)
        )
        std_obj = float(
            np.std(fold_scores)
        )

        obj = (
            0.65 * mean_obj
            + 0.30 * min_obj
            - 0.05 * std_obj
        )

        rec = {
            "objective": float(obj),
            "global_t": float(global_t),
            "alpha": float(alpha),
            "shift": float(shift),
            "folds": fold_metrics,
        }

        if (
            best is None
            or rec["objective"]
            > best["objective"]
        ):
            best = rec

    if best is None:
        raise RuntimeError(
            "No valid LODO threshold candidate"
        )

    attr_all = attribute_ma_thresholds(
        y_true,
        probs,
    )

    thresholds = (
        (1.0 - best["alpha"])
        * best["global_t"]
        + best["alpha"]
        * attr_all
        + best["shift"]
    )

    thresholds = np.clip(
        thresholds,
        0.34,
        0.88,
    ).astype(np.float32)

    final = evaluate_probs(
        y_true,
        probs,
        thresholds,
    )

    report = dict(best)
    report["final"] = final

    return thresholds, report


def main():
    args = parse_args()
    seed_everything(2027)

    seeds = parse_members(
        args.members
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 100, flush=True)
    print(
        "UPAR PROTOTYPE PRECISION RECOVERY CALIBRATION",
        flush=True,
    )
    print(
        "Checkpoint prefix:",
        args.checkpoint_prefix,
        flush=True,
    )
    print(
        "Seeds:",
        seeds,
        flush=True,
    )
    print(
        "Gate max:",
        args.gate_max,
        flush=True,
    )
    print("=" * 100, flush=True)

    loader = build_val_loader(
        args
    )

    member_records = []
    adjusted_logits = []

    y_true = None
    domains = None

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        path = checkpoint_path(
            args.checkpoint_dir,
            args.checkpoint_prefix,
            seed,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

        ck = safe_torch_load(
            path,
            map_location="cpu",
        )

        state = (
            ck.get(
                "inference_model_state"
            )
            or ck.get(
                "ema_model_state"
            )
            or ck.get(
                "model_state"
            )
        )

        if state is None:
            raise RuntimeError(
                f"{path} contains no model state"
            )

        config = dict(
            ck.get(
                "prototype_config",
                {},
            )
        )

        # Old checkpoints do not contain the new gate range keys.
        config.setdefault(
            "prototype_gate_min",
            0.08,
        )
        config.setdefault(
            "prototype_gate_max",
            0.92,
        )

        model = HybridPrototypePAR(
            pretrained=False,
            **config,
        )

        model.load_state_dict(
            state,
            strict=True,
        )

        model = model.to(
            device
        ).eval()

        print(
            f"\n[{index}/{len(seeds)}] "
            f"seed={seed} | {path.name}",
            flush=True,
        )

        (
            linear_logits,
            proto_logits,
            learned_gate,
            yt,
            ds,
        ) = predict_components(
            model,
            loader,
            device,
            use_flip=args.tta == "flip",
        )

        if y_true is None:
            y_true = yt
            domains = ds
        else:
            if not np.array_equal(
                y_true,
                yt,
            ):
                raise RuntimeError(
                    "Validation target order changed"
                )
            if not np.array_equal(
                domains,
                ds,
            ):
                raise RuntimeError(
                    "Validation domain order changed"
                )

        print(
            "  learned gate:",
            {
                "min": float(learned_gate.min()),
                "mean": float(learned_gate.mean()),
                "max": float(learned_gate.max()),
            },
            flush=True,
        )

        gate_result, member_logits = optimize_group_scales(
            y_true,
            linear_logits,
            proto_logits,
            learned_gate,
            domains,
            gate_max=float(args.gate_max),
            coordinate_passes=int(args.coordinate_passes),
        )

        print(
            "  selected group scales:",
            gate_result["group_scales"],
            flush=True,
        )
        print(
            "  gate-search metrics:",
            json.dumps(
                gate_result["overall"],
                indent=2,
            ),
            flush=True,
        )

        adjusted_logits.append(
            member_logits
        )

        member_records.append({
            "model": "prototype_convnext_tiny",
            "seed": int(seed),
            "checkpoint": str(path),
            "checkpoint_prefix": args.checkpoint_prefix,
            "prototype_config": config,
            "learned_gate_stats": {
                "min": float(learned_gate.min()),
                "mean": float(learned_gate.mean()),
                "max": float(learned_gate.max()),
            },
            "group_scales": gate_result["group_scales"],
            "gate_max": float(args.gate_max),
            "gate_search_objective": float(
                gate_result["objective"]
            ),
            "gate_search_metrics": gate_result["overall"],
        })

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Ensemble weight search after gate correction.
    # ----------------------------------------------------------
    ensemble_candidates = []

    for weights in tqdm(
        ensemble_weight_candidates(
            len(adjusted_logits)
        ),
        desc="Post-gate ensemble-weight search",
        unit="candidate",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
    ):
        logits = ensemble_logits(
            adjusted_logits,
            weights,
        )

        op = fast_operating_point(
            y_true,
            logits,
            domains,
        )

        ensemble_candidates.append({
            "weights": weights,
            "objective": float(
                op["objective"]
            ),
            "overall": op["overall"],
            "robust": float(
                op["robust"]
            ),
        })

    ensemble_candidates.sort(
        key=lambda x: x["objective"],
        reverse=True,
    )

    best_weights = ensemble_candidates[0][
        "weights"
    ]

    print(
        "\nBest ensemble weights:",
        best_weights.tolist(),
        flush=True,
    )

    final_logits = ensemble_logits(
        adjusted_logits,
        best_weights,
    )

    final_probs = sigmoid_np(
        final_logits
    ).astype(np.float32)

    thresholds, threshold_report = tune_precision_lodo_thresholds(
        y_true,
        final_probs,
        domains,
    )

    robust, overall, per_domain = robust_selection_score(
        y_true,
        final_probs,
        thresholds,
        domains,
    )

    print(
        "\nFINAL PRECISION-RECOVERY VALIDATION",
        flush=True,
    )
    print(
        json.dumps(
            overall,
            indent=2,
        ),
        flush=True,
    )

    print(
        "Robust:",
        robust,
        flush=True,
    )

    for d, metrics in per_domain.items():
        print(
            DOMAIN_NAMES.get(
                int(d),
                str(d),
            ),
            json.dumps(
                metrics,
                indent=2,
            ),
            flush=True,
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "format": "upar-prototype-precision-calibration",
            "members": member_records,
            "weights": torch.tensor(
                best_weights,
                dtype=torch.float32,
            ),
            "thresholds": torch.tensor(
                thresholds,
                dtype=torch.float32,
            ),
            "threshold_logits": torch.tensor(
                logit_np(thresholds),
                dtype=torch.float32,
            ),
            "tta": args.tta,
            "gate_max": float(args.gate_max),
            "metrics": overall,
            "robust": float(robust),
            "per_domain": per_domain,
            "threshold_report": threshold_report,
            "attribute_groups": ATTRIBUTE_GROUPS,
            "attribute_names": ATTRIBUTE_NAMES,
            "image_height": 288,
            "image_width": 144,
        },
        args.output,
    )

    print(
        "\nSaved:",
        args.output,
        flush=True,
    )


if __name__ == "__main__":
    main()
