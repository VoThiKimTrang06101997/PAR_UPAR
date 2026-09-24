from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

from .config import Config, ATTRIBUTE_NAMES
from .data import TemplateInferenceDataset, EvalCollator
from .model import VAPORPAR
from .checkpoint import load_checkpoint
from .calibration import apply_calibration
from .utils import move_inputs, get_amp_dtype

def validate_submission(df, template_df=None):
    if df.shape[1] != 41:
        raise ValueError(f"Expected 41 columns (image + 40 attributes), got {df.shape[1]}")
    if list(df.columns[1:]) != ATTRIBUTE_NAMES:
        raise ValueError("Attribute column order does not match official UPAR vocabulary.")
    vals = df.iloc[:,1:].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("Submission contains NaN/Inf.")
    if vals.min() < 0 or vals.max() > 1:
        raise ValueError("Submission probabilities must be in [0,1].")
    if template_df is not None:
        if len(df) != len(template_df):
            raise ValueError("Submission row count differs from template.")
        if not np.array_equal(df.iloc[:,0].astype(str).values, template_df.iloc[:,0].astype(str).values):
            raise ValueError("Submission image order differs from template.")
    return True

@torch.no_grad()
def run_inference_from_template(cfg: Config, checkpoint_path, template_csv, image_root,
                                output_csv=None, output_zip=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = VAPORPAR(cfg).to(device)
    model.initialize_prompts(processor, device)
    ck = load_checkpoint(checkpoint_path, model, device="cpu")
    calibration = ck.get("calibration")
    model.eval()

    ds = TemplateInferenceDataset(template_csv, image_root)
    loader = DataLoader(
        ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=EvalCollator(processor, has_targets=False)
    )

    amp_dtype = get_amp_dtype()
    probs_all, paths = [], []
    for batch in tqdm(loader, desc="Inference"):
        inputs = move_inputs(batch["inputs"], device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(cfg.amp and device.type=="cuda")):
            out = model(inputs)
            logits = apply_calibration(out["logits"], calibration)
            probs = torch.sigmoid(logits)
        probs_all.append(probs.float().cpu().numpy())
        paths.extend(batch["paths"])

    probs = np.concatenate(probs_all, 0)
    template = pd.read_csv(template_csv)
    image_col = template.columns[0]

    # Preserve exact organizer image-column name and row order; enforce official 40-attribute order.
    out_df = pd.DataFrame({image_col: paths})
    for j,c in enumerate(ATTRIBUTE_NAMES):
        out_df[c] = probs[:,j]

    validate_submission(out_df, template_df=template)
    output_csv = Path(output_csv or (Path(cfg.results_dir)/"predictions.csv"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, float_format="%.8f")

    output_zip = Path(output_zip or (Path(cfg.results_dir)/"submission_task1.zip"))
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_csv, arcname="predictions.csv")

    # Verify ZIP root structure.
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        if names != ["predictions.csv"]:
            raise RuntimeError(f"Unexpected submission ZIP structure: {names}")

    print("Saved CSV:", output_csv)
    print("Saved ZIP:", output_zip)
    return out_df, str(output_zip)

def make_dev_format_check(cfg: Config, checkpoint_path):
    """Use official validation gt.csv only as an image-order template for a local format smoke test."""
    return run_inference_from_template(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        template_csv=cfg.val_csv,
        image_root=cfg.data_dir,
        output_csv=Path(cfg.results_dir)/"dev_predictions.csv",
        output_zip=Path(cfg.results_dir)/"dev_submission_format_check.zip",
    )
