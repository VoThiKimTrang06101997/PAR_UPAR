from vapor_par.config import Config
from vapor_par.inference import make_dev_format_check

if __name__ == "__main__":
    cfg = Config(
        checkpoint_dir="/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints",
        results_dir="/content/drive/MyDrive/Pedestrian Attribute Recognition/Results",
    )
    ckpt = "/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints/best_calibrated.pt"
    df, zip_path = make_dev_format_check(cfg, ckpt)
    print("Development format-check ZIP:", zip_path)
    print(df.head())
