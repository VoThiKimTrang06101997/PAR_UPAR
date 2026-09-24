import argparse
from vapor_par.config import Config
from vapor_par.inference import run_inference_from_template

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, help="Official Task-1 submission template CSV")
    p.add_argument("--image-root", required=True, help="Root directory containing private-test images")
    p.add_argument(
        "--checkpoint",
        default="/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints/best_calibrated.pt"
    )
    p.add_argument(
        "--output-csv",
        default="/content/drive/MyDrive/Pedestrian Attribute Recognition/Results/predictions.csv"
    )
    p.add_argument(
        "--output-zip",
        default="/content/drive/MyDrive/Pedestrian Attribute Recognition/Results/submission_task1.zip"
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        checkpoint_dir="/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints",
        results_dir="/content/drive/MyDrive/Pedestrian Attribute Recognition/Results",
    )
    _, zip_path = run_inference_from_template(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        template_csv=args.template,
        image_root=args.image_root,
        output_csv=args.output_csv,
        output_zip=args.output_zip,
    )
    print("READY TO SUBMIT:", zip_path)
