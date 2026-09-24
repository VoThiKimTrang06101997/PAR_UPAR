from pathlib import Path
import argparse
from vapor_par.checkpoint import load_checkpoint_payload


def _score_checkpoint(ck):
    return (
        int(ck.get("global_step", -1)),
        int(ck.get("epoch", -1)),
        int(ck.get("next_batch_idx", -1)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-dir",
        default="/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints",
    )
    args = p.parse_args()

    d = Path(args.checkpoint_dir)
    candidates = [
        d / "resume_step.pt",
        d / "last.pt",
        d / "best.pt",
        d / "best_calibrated.pt",
    ]
    existing = [x for x in candidates if x.exists()]

    if not existing:
        print("No checkpoint found in", d)
        return

    payloads = {}

    for path in existing:
        ck = load_checkpoint_payload(path, device="cpu")
        payloads[path] = ck
        cfg = ck.get("cfg") or {}
        data = ck.get("data_state") or {}

        print("\n===", path.name, "===")
        print("epoch          :", ck.get("epoch"))
        print("next_batch_idx :", ck.get("next_batch_idx"))
        print("global_step    :", ck.get("global_step"))
        print("best_score     :", ck.get("best_score"))
        print("batch_size     :", cfg.get("batch_size"))
        print("seed           :", cfg.get("seed"))
        print("dataset_size   :", data.get("dataset_size"))
        print("total_batches  :", data.get("total_batches"))

        if data.get("total_batches"):
            b = int(ck.get("next_batch_idx", 0))
            t = int(data["total_batches"])
            print(
                "resume_percent :",
                f"{100*b/max(t,1):.2f}%"
            )

    resume_candidates = [
        (path, payloads[path])
        for path in (d / "resume_step.pt", d / "last.pt")
        if path in payloads
    ]

    if resume_candidates:
        resume_candidates.sort(
            key=lambda x: _score_checkpoint(x[1]),
            reverse=True,
        )
        path, ck = resume_candidates[0]
        score = _score_checkpoint(ck)

        print("\n=== RECOMMENDED RESUME ===")
        print("file           :", path.name)
        print("global_step    :", score[0])
        print("epoch          :", score[1])
        print("next_batch_idx :", score[2])


if __name__ == "__main__":
    main()
