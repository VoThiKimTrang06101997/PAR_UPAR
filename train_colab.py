import argparse
from pathlib import Path
from vapor_par.config import Config
from vapor_par.train import run_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--eval-batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-every-steps", type=int, default=100)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing resume_step.pt/last.pt and start a new run. Default is RESUME.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        repo_dir="/content/UPAR-Challenge-2027",
        data_dir="/content/UPAR-Challenge-2027/data",
        train_csv="/content/UPAR-Challenge-2027/data/annotations/task1/train/gt.csv",
        val_csv="/content/UPAR-Challenge-2027/data/annotations/task1/val/gt.csv",
        checkpoint_dir="/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints",
        results_dir="/content/drive/MyDrive/Pedestrian Attribute Recognition/Results",
        model_id="google/siglip2-base-patch16-naflex",
        hf_cache_dir="/content/drive/MyDrive/PedestrianAttributeRecognition/HFCache",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    cfg.ensure_dirs()

    ckpt_dir = Path(cfg.checkpoint_dir)
    if not args.restart:
        if (ckpt_dir / "resume_step.pt").exists():
            print("[AUTO-RESUME] Found resume_step.pt")
        elif (ckpt_dir / "last.pt").exists():
            print("[AUTO-RESUME] Found last.pt")
        else:
            print("[AUTO-RESUME] No previous checkpoint found; starting a new run.")
    else:
        print("[RESTART] Existing checkpoints will be ignored for this run.")

    state = run_training(cfg, resume=(not args.restart))
    print("Best calibrated checkpoint:", state["best_checkpoint"])
    print("Final validation metrics:", state["metrics"])
