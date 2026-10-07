from __future__ import annotations

import argparse
import json
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
from vapor_par.metrics import (
    robust_selection_score,
    tune_lodo_thresholds,
)


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
        default=Path(
            "/content/drive/MyDrive/"
            "PedestrianAttributeRecognition/Checkpoints"
        ),
    )
    p.add_argument(
        "--members",
        type=str,
        required=True,
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--tta",
        choices=["none", "flip"],
        default="flip",
    )
    return p.parse_args()


def parse_members(spec):
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No prototype ensemble seeds specified")
    return out


def checkpoint_path(root, seed):
    return root / f"prototype_convnext_tiny_seed{seed}_best.pt"


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


@torch.inference_mode()
def predict_logits(model, loader, device, flip=True):
    model.eval()
    zs, ys, ds = [], [], []

    for batch in tqdm(
        loader,
        desc="Validation inference + flip-TTA" if flip else "Validation inference",
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
            z = model(x)
            if flip:
                z = 0.5 * (
                    z
                    + model(torch.flip(x, dims=[3]))
                )

        zs.append(z.float().cpu().numpy())

        y = batch["target"].numpy()
        v = batch["valid"].numpy()
        ys.append(np.where(v > 0, y, -1).astype(np.int32))
        ds.append(np.asarray(batch["domain"], dtype=np.int64))

    return (
        np.concatenate(zs),
        np.concatenate(ys),
        np.concatenate(ds),
    )


def build_val_loader(args):
    data_root = args.repo_root / "data"
    val_csv = (
        data_root
        / "annotations"
        / "task1"
        / "val"
        / "gt.csv"
    )

    if not val_csv.exists():
        raise FileNotFoundError(val_csv)

    df = pd.read_csv(val_csv)
    resolver = ImageResolver(data_root, args.repo_root)

    ds = UPARDataset(
        df,
        resolver,
        build_eval_transform(288, 144),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print("Validation:", df.shape, flush=True)
    return loader


def weight_candidates(n_models):
    if n_models == 1:
        return [np.asarray([1.0], dtype=np.float64)]

    if n_models == 2:
        return [
            np.asarray([w, 1.0 - w], dtype=np.float64)
            for w in np.arange(0.0, 1.0001, 0.10)
        ]

    out = []
    for i in range(n_models):
        w = np.zeros(n_models, dtype=np.float64)
        w[i] = 1.0
        out.append(w)

    out.append(
        np.full(
            n_models,
            1.0 / n_models,
            dtype=np.float64,
        )
    )
    return out


def ensemble_logits(logits_list, weights):
    out = np.zeros_like(logits_list[0], dtype=np.float64)
    for w, z in zip(weights, logits_list):
        out += float(w) * z.astype(np.float64)
    return out


def objective(overall, robust, per_domain):
    domain_scores = [
        float(v["Challenge_Avg"])
        for v in per_domain.values()
    ]

    min_domain = (
        min(domain_scores)
        if domain_scores
        else float(overall["Challenge_Avg"])
    )

    weak_component = min(
        float(overall["mA"]),
        float(overall["instance_f1"]),
    )

    return float(
        0.35 * float(overall["Challenge_Avg"])
        + 0.35 * float(robust)
        + 0.15 * float(min_domain)
        + 0.15 * float(weak_component)
    )


def main():
    args = parse_args()
    seed_everything(2027)

    members = parse_members(args.members)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100, flush=True)
    print("UPAR PROTOTYPE AUTOENSEMBLE CALIBRATION", flush=True)
    print("Members:", members, flush=True)
    print("Device :", device, flush=True)
    print("TTA    :", args.tta, flush=True)
    print("=" * 100, flush=True)

    loader = build_val_loader(args)

    logits_list = []
    member_info = []
    y_true = None
    domains = None

    for idx, seed in enumerate(members, start=1):
        p = checkpoint_path(
            args.checkpoint_dir,
            seed,
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {p}"
            )

        ck = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        state = (
            ck.get("inference_model_state")
            or ck.get("ema_model_state")
            or ck.get("model_state")
        )

        if state is None:
            raise RuntimeError(
                f"{p} has no inference model state"
            )

        local_metrics = ck.get("metrics_calibrated", {})
        local_score = float(
            local_metrics.get(
                "Challenge_Avg",
                ck.get(
                    "selection_calibrated_robust",
                    ck.get("selection", 0.0),
                ),
            )
        )

        print(
            f"\n[{idx}/{len(members)}] "
            f"prototype_convnext_tiny:{seed} "
            f"| local calibrated={local_score:.6f}",
            flush=True,
        )

        proto_cfg = dict(ck.get("prototype_config", {}))
        # gate init only affects construction; checkpoint state overwrites it.
        model = HybridPrototypePAR(
            pretrained=False,
            **proto_cfg,
        )
        model.load_state_dict(
            state,
            strict=True,
        )
        model = model.to(device).eval()

        logits, yt, ds = predict_logits(
            model,
            loader,
            device,
            flip=args.tta == "flip",
        )

        if y_true is None:
            y_true = yt
            domains = ds
        else:
            if not np.array_equal(y_true, yt):
                raise RuntimeError(
                    "Validation target order mismatch"
                )
            if not np.array_equal(domains, ds):
                raise RuntimeError(
                    "Validation domain order mismatch"
                )

        logits_list.append(
            logits.astype(np.float32)
        )

        member_info.append({
            "model": "prototype_convnext_tiny",
            "seed": int(seed),
            "checkpoint": str(p),
            "checkpoint_local_score": local_score,
            "checkpoint_source": ck.get("source"),
            "prototype_config": proto_cfg,
        })

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Coarse raw-logit ensemble search.
    # Includes 100/0 and 0/100 endpoints.
    # ---------------------------------------------------------
    coarse = []

    for weights in tqdm(
        weight_candidates(len(members)),
        desc="Prototype ensemble-weight search",
        unit="candidate",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        z = ensemble_logits(
            logits_list,
            weights,
        )

        probs = sigmoid_np(z).astype(np.float32)

        robust, overall, per_domain = (
            robust_selection_score(
                y_true,
                probs,
                0.5,
                domains,
            )
        )

        coarse.append({
            "weights": weights,
            "objective": objective(
                overall,
                robust,
                per_domain,
            ),
            "overall": overall,
            "robust": float(robust),
        })

    coarse.sort(
        key=lambda x: x["objective"],
        reverse=True,
    )

    print("\nTop coarse weights:", flush=True)

    for item in coarse[:5]:
        print(
            [round(float(x), 3) for x in item["weights"]],
            "obj=",
            round(float(item["objective"]), 6),
            "score=",
            round(
                float(
                    item["overall"]["Challenge_Avg"]
                ),
                6,
            ),
            flush=True,
        )

    # Top candidates + explicit single-model endpoints.
    selected = coarse[:4]

    selected_keys = {
        tuple(x["weights"].tolist())
        for x in selected
    }

    for i in range(len(members)):
        endpoint = np.zeros(
            len(members),
            dtype=np.float64,
        )
        endpoint[i] = 1.0
        key = tuple(endpoint.tolist())

        if key not in selected_keys:
            for item in coarse:
                if tuple(item["weights"].tolist()) == key:
                    selected.append(item)
                    selected_keys.add(key)
                    break

    # ---------------------------------------------------------
    # Calibrate thresholds AFTER ensemble.
    # Search shrinkage toward 0.5 + very small global bias.
    # ---------------------------------------------------------
    best = None

    for item in tqdm(
        selected,
        desc="LODO post-ensemble calibration",
        unit="candidate",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        weights = item["weights"]

        z = ensemble_logits(
            logits_list,
            weights,
        )

        probs = sigmoid_np(z).astype(np.float32)

        base_thresholds, lodo_report = (
            tune_lodo_thresholds(
                y_true,
                probs,
                domains,
            )
        )

        base_logits = logit_np(
            base_thresholds
        )

        for shrink in (
            0.65,
            0.80,
            0.95,
            1.00,
        ):
            for bias in (
                -0.03,
                0.00,
                0.03,
                0.06,
            ):
                threshold_logits = (
                    shrink * base_logits
                    + bias
                )

                thresholds = sigmoid_np(
                    threshold_logits
                ).astype(np.float32)

                robust, overall, per_domain = (
                    robust_selection_score(
                        y_true,
                        probs,
                        thresholds,
                        domains,
                    )
                )

                obj = objective(
                    overall,
                    robust,
                    per_domain,
                )

                # Mild regularization against aggressive source calibration.
                obj -= (
                    0.002 * abs(float(bias))
                    + 0.001 * abs(1.0 - float(shrink))
                )

                rec = {
                    "objective": float(obj),
                    "weights": weights.copy(),
                    "threshold_logits": threshold_logits.astype(np.float32),
                    "thresholds": thresholds,
                    "overall": overall,
                    "robust": float(robust),
                    "per_domain": per_domain,
                    "shrink": float(shrink),
                    "bias": float(bias),
                    "lodo_report": lodo_report,
                }

                if (
                    best is None
                    or rec["objective"] > best["objective"]
                ):
                    best = rec

    if best is None:
        raise RuntimeError(
            "No valid ensemble calibration found"
        )

    print("\n" + "=" * 100, flush=True)
    print("BEST PROTOTYPE AUTOENSEMBLE", flush=True)

    for member, weight in zip(
        member_info,
        best["weights"],
    ):
        print(
            f"{member['model']}:{member['seed']} "
            f"weight={float(weight):.4f} "
            f"local={member['checkpoint_local_score']:.6f}",
            flush=True,
        )

    print(
        "Threshold shrink:",
        best["shrink"],
        flush=True,
    )
    print(
        "Threshold bias  :",
        best["bias"],
        flush=True,
    )
    print(
        "Objective       :",
        best["objective"],
        flush=True,
    )
    print(
        "Robust          :",
        best["robust"],
        flush=True,
    )
    print(
        json.dumps(
            best["overall"],
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
            "format": "upar-prototype-autoensemble-calibration",
            "members": member_info,
            "weights": torch.tensor(
                best["weights"],
                dtype=torch.float32,
            ),
            "threshold_logits": torch.tensor(
                best["threshold_logits"],
                dtype=torch.float32,
            ),
            "thresholds": torch.tensor(
                best["thresholds"],
                dtype=torch.float32,
            ),
            "tta": args.tta,
            "objective": float(best["objective"]),
            "robust": float(best["robust"]),
            "metrics": best["overall"],
            "per_domain": best["per_domain"],
            "threshold_shrink": best["shrink"],
            "threshold_bias": best["bias"],
            "lodo_report": best["lodo_report"],
            "attribute_names": ATTRIBUTE_NAMES,
            "image_height": 288,
            "image_width": 144,
        },
        args.output,
    )

    print(
        "\nSaved calibration:",
        args.output,
        flush=True,
    )


if __name__ == "__main__":
    main()
