from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--repo-dir", default="/content/PAR_UPAR")
    p.add_argument(
        "--checkpoint",
        default=(
            "/content/drive/MyDrive/PedestrianAttributeRecognition/"
            "Checkpoints/best_calibrated.pt"
        ),
    )
    p.add_argument(
        "--output",
        default=(
            "/content/drive/MyDrive/Pedestrian Attribute Recognition/"
            "Results/PAR_UPAR_submission.zip"
        ),
    )

    p.add_argument(
        "--metadata-yaml",
        default="",
        help=(
            "Path to metadata.yaml copied from the official example submission. "
            "If omitted, the builder searches --official-starter-dir."
        ),
    )
    p.add_argument(
        "--official-starter-dir",
        default="/content/UPAR-Challenge-2027",
    )

    p.add_argument(
        "--model-id",
        default="google/siglip2-base-patch16-naflex",
    )
    p.add_argument("--num-heads", type=int, default=8)

    # Transformers 5.2.0 is documented for PyTorch 2.4+ and is therefore a
    # better target for the challenge's torch 2.4.1 runtime than training's
    # latest 5.17 stack.
    p.add_argument("--vendor-transformers", default="5.2.0")

    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def locate_metadata_yaml(explicit_path, starter_dir):
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"metadata.yaml not found: {p}")
        return p

    starter = Path(starter_dir)
    candidates = []

    if starter.exists():
        for p in starter.rglob("metadata.yaml"):
            low = str(p).lower()

            # Avoid Codabench ingestion/scoring metadata. We need the metadata
            # that belongs to the participant example submission.
            if "ingestion_program" in low or "scoring_program" in low:
                continue

            score = 0
            if (p.parent / "run.py").exists():
                score += 100
            if "example" in low:
                score += 30
            if "submission" in low:
                score += 30
            if "starter" in low:
                score += 10

            if score > 0:
                candidates.append((score, len(str(p)), p))

    if not candidates:
        raise FileNotFoundError(
            "Could not find the participant example-submission metadata.yaml. "
            "Download the example submission from the challenge Data page and "
            "pass its metadata.yaml with --metadata-yaml. The builder will copy "
            "that file unchanged."
        )

    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen = candidates[0][2]
    print("Using official metadata.yaml:", chosen)
    return chosen


def copy_metadata_unchanged(source, destination):
    before = sha256(source)
    shutil.copyfile(source, destination)
    after = sha256(destination)

    if before != after:
        raise RuntimeError("metadata.yaml changed while copying.")

    print("metadata.yaml copied unchanged")
    print("metadata sha256:", before)


def make_full_offline_bundle(
    repo_dir,
    checkpoint_path,
    model_id,
    num_heads,
    out_path,
):
    sys.path.insert(0, str(repo_dir))

    from transformers import AutoProcessor
    from vapor_par.config import Config, ATTRIBUTE_NAMES
    from vapor_par.model import VAPORPAR
    from vapor_par.checkpoint import (
        load_checkpoint_payload,
        apply_checkpoint_payload,
    )

    ckpt = load_checkpoint_payload(checkpoint_path, device="cpu")
    saved_cfg = ckpt.get("cfg") or {}

    model_id = saved_cfg.get("model_id", model_id)
    num_heads = int(saved_cfg.get("num_heads", num_heads))
    num_attributes = int(saved_cfg.get("num_attributes", 40))

    if num_attributes != 40:
        raise RuntimeError(
            f"Challenge requires 40 attributes; checkpoint has {num_attributes}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cfg = Config(
        model_id=model_id,
        num_heads=num_heads,
        num_attributes=num_attributes,
        hf_cache_dir=saved_cfg.get(
            "hf_cache_dir",
            "/content/drive/MyDrive/PedestrianAttributeRecognition/HFCache",
        ),
    )

    print("Reconstructing final trained model from:", checkpoint_path)
    print("Base VLM:", model_id)

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=(saved_cfg.get("hf_cache_dir") or None),
    )
    model = VAPORPAR(cfg).to(device)
    model.initialize_prompts(processor, device)

    apply_checkpoint_payload(
        ckpt,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        restore_rng=False,
    )
    model.eval()

    # Training checkpoints are deliberately compact. Export the COMPLETE final
    # vision encoder here while Colab has the pretrained model available.
    vision_state = {
        k: v.detach().cpu()
        for k, v in model.vlm.vision_model.state_dict().items()
    }

    full_state = model.state_dict()
    head_state = {
        k: v.detach().cpu()
        for k, v in full_state.items()
        if not k.startswith("vlm.")
    }

    image_processor = getattr(
        processor,
        "image_processor",
        processor,
    )

    bundle = {
        "format_version": 3,
        "model_name": "VAPOR-PAR",
        "source_model_id": model_id,
        "attribute_names": list(ATTRIBUTE_NAMES),
        "vision_config": model.vlm.vision_model.config.to_dict(),
        "processor_config": image_processor.to_dict(),
        "vision_state": vision_state,
        "head_state": head_state,
        "calibration": ckpt.get("calibration"),
        "thresholds": ckpt.get("thresholds"),
        "meta": {
            "hidden_dim": int(model.hidden_dim),
            "text_dim": int(model.text_dim),
            "num_attributes": int(model.num_attributes),
            "num_heads": int(cfg.num_heads),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "checkpoint_global_step": int(
                ckpt.get("global_step", -1)
            ),
            "best_score": float(ckpt.get("best_score", -1.0)),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)

    print(
        "Offline inference bundle:",
        out_path,
        f"({out_path.stat().st_size / 1024**2:.1f} MiB)",
    )


def download_runtime_wheels(work_dir, transformers_version):
    wheel_dir = work_dir / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)

    # Download for the challenge target, not for the current Colab Python.
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheel_dir),
        "--only-binary=:all:",
        "--platform",
        "manylinux2014_x86_64",
        "--python-version",
        "311",
        "--implementation",
        "cp",
        "--abi",
        "cp311",
        f"transformers=={transformers_version}",
    ]

    print("Downloading Python 3.11 offline runtime wheels...")
    subprocess.check_call(cmd)
    return wheel_dir


def unpack_vendor(wheel_dir, vendor_dir):
    vendor_dir.mkdir(parents=True, exist_ok=True)

    # These are guaranteed by the challenge base image and should not be
    # replaced by foreign ABI wheels from the builder machine.
    skip_prefixes = (
        "numpy-",
        "torch-",
        "torchvision-",
        "pillow-",
        "nvidia_",
        "triton-",
    )

    kept = []
    skipped = []

    for wheel in sorted(wheel_dir.glob("*.whl")):
        name = wheel.name.lower()

        if name.startswith(skip_prefixes):
            skipped.append(wheel.name)
            continue

        with zipfile.ZipFile(wheel, "r") as zf:
            zf.extractall(vendor_dir)

        kept.append(wheel.name)

    print("Vendored runtime wheels:")
    for name in kept:
        print(" +", name)

    if skipped:
        print("Base-image packages not vendored:")
        for name in skipped:
            print(" -", name)


def copy_runtime_sources(submission_dir, runtime_src_dir):
    for name in ("run.py", "model_runtime.py"):
        shutil.copy2(
            runtime_src_dir / name,
            submission_dir / name,
        )


def clean_tree(root):
    for p in list(root.rglob("__pycache__")):
        shutil.rmtree(p, ignore_errors=True)

    for p in list(root.rglob("*.pyc")):
        p.unlink(missing_ok=True)

    for p in list(root.rglob(".DS_Store")):
        p.unlink(missing_ok=True)


def create_zip(submission_dir, output_zip):
    clean_tree(submission_dir)

    output_zip.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_zip.unlink(missing_ok=True)

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for p in submission_dir.rglob("*"):
            if not p.is_file():
                continue

            rel = p.relative_to(submission_dir)

            # Equivalent to:
            # zip -r ../submission.zip . -x */__pycache__/* *.DS_Store .git/*
            if "__pycache__" in rel.parts:
                continue
            if ".git" in rel.parts:
                continue
            if p.name == ".DS_Store":
                continue

            # Zip CONTENTS, not the containing folder.
            zf.write(p, arcname=str(rel))

    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()

    required = {
        "run.py",
        "metadata.yaml",
        "model_runtime.py",
        "weights/inference_bundle.pt",
    }
    missing = sorted(required - set(names))

    if missing:
        raise RuntimeError(
            f"Invalid submission ZIP. Missing: {missing}"
        )

    if any(x.startswith("submission/") for x in names):
        raise RuntimeError(
            "Submission folder itself was zipped. "
            "Zip its CONTENTS instead."
        )

    if not any(x.startswith("vendor/transformers/") for x in names):
        raise RuntimeError(
            "Offline transformers runtime is missing."
        )

    print("Submission ZIP root validation: OK")
    print("ZIP size:", f"{output_zip.stat().st_size / 1024**2:.1f} MiB")
    print("READY TO UPLOAD:", output_zip)


def main():
    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_zip = Path(args.output).resolve()

    runtime_src = (
        Path(__file__).resolve().parent
        / "submission_runtime"
    )

    if not repo_dir.exists():
        raise FileNotFoundError(
            f"PAR_UPAR repo not found: {repo_dir}"
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train/calibrate first so best_calibrated.pt exists."
        )

    metadata_source = locate_metadata_yaml(
        args.metadata_yaml,
        args.official_starter_dir,
    )

    work = Path(
        tempfile.mkdtemp(
            prefix="par_upar_submission_"
        )
    )

    try:
        submission = work / "submission"
        weights = submission / "weights"
        weights.mkdir(
            parents=True,
            exist_ok=True,
        )

        copy_runtime_sources(
            submission,
            runtime_src,
        )

        copy_metadata_unchanged(
            metadata_source,
            submission / "metadata.yaml",
        )

        make_full_offline_bundle(
            repo_dir=repo_dir,
            checkpoint_path=checkpoint,
            model_id=args.model_id,
            num_heads=args.num_heads,
            out_path=weights / "inference_bundle.pt",
        )

        wheels = download_runtime_wheels(
            work,
            args.vendor_transformers,
        )
        unpack_vendor(
            wheels,
            submission / "vendor",
        )

        manifest = {
            "challenge": "UPAR Challenge 2027 Track 1",
            "api": {
                "load_model": True,
                "predict_image": True,
                "predict_batch": True,
                "batch_size": 64,
                "output_attributes": 40,
            },
            "offline": True,
            "runtime": {
                "python": "3.11",
                "torch": "2.4.1",
                "cuda": "12.1",
            },
            "vendored_transformers": args.vendor_transformers,
            "checkpoint": checkpoint.name,
            "metadata_sha256": sha256(
                submission / "metadata.yaml"
            ),
        }

        (
            submission / "submission_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        create_zip(
            submission,
            output_zip,
        )

    finally:
        shutil.rmtree(
            work,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
