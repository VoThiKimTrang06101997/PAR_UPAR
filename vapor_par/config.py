from dataclasses import dataclass, asdict
from pathlib import Path

ATTRIBUTE_NAMES = [
    "Age-Young","Age-Adult","Age-Old","Gender-Female",
    "Hair-Length-Short","Hair-Length-Long","Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black","UpperBody-Color-Blue","UpperBody-Color-Brown",
    "UpperBody-Color-Green","UpperBody-Color-Grey","UpperBody-Color-Orange",
    "UpperBody-Color-Pink","UpperBody-Color-Purple","UpperBody-Color-Red",
    "UpperBody-Color-White","UpperBody-Color-Yellow","UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black","LowerBody-Color-Blue","LowerBody-Color-Brown",
    "LowerBody-Color-Green","LowerBody-Color-Grey","LowerBody-Color-Orange",
    "LowerBody-Color-Pink","LowerBody-Color-Purple","LowerBody-Color-Red",
    "LowerBody-Color-White","LowerBody-Color-Yellow","LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts","LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack","Accessory-Bag","Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun","Accessory-Hat"
]

ONTOLOGY_GROUPS = {
    "age": [0,1,2],
    "hair": [4,5,6],
    "upper_color": list(range(8,20)),
    "lower_color": list(range(21,33)),
    "lower_type": [33,34],
}

DOMAIN_TO_ID = {"Market1501": 0, "PA100k": 1, "PETA": 2}

@dataclass
class Config:
    # Official UPAR 2027 repository/data
    repo_dir: str = "/content/UPAR-Challenge-2027"
    data_dir: str = "/content/UPAR-Challenge-2027/data"
    train_csv: str = "/content/UPAR-Challenge-2027/data/annotations/task1/train/gt.csv"
    val_csv: str = "/content/UPAR-Challenge-2027/data/annotations/task1/val/gt.csv"

    # User-requested Google Drive outputs
    checkpoint_dir: str = "/content/drive/MyDrive/PedestrianAttributeRecognition/Checkpoints"
    results_dir: str = "/content/drive/MyDrive/Pedestrian Attribute Recognition/Results"

    # VLM
    model_id: str = "google/siglip2-base-patch16-naflex"
    hf_cache_dir: str = "/content/drive/MyDrive/PedestrianAttributeRecognition/HFCache"
    num_attributes: int = 40
    num_heads: int = 8
    dropout: float = 0.10
    temperature: float = 0.07
    max_text_length: int = 64

    # Training
    seed: int = 605
    batch_size: int = 12
    eval_batch_size: int = 24
    num_workers: int = 2
    epochs: int = 25
    lr_head: float = 3e-4
    lr_backbone: float = 5e-6
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 5.0
    accumulation_steps: int = 1
    amp: bool = True

    # Curriculum
    semantic_warmup_epochs: int = 3
    consistency_start_epoch: int = 5
    unfreeze_start_epoch: int = 15
    unfreeze_last_n_vision_blocks: int = 2

    # Loss weights
    w_bal: float = 1.0
    w_sem: float = 0.20
    w_dg: float = 0.10
    w_anchor: float = 0.10
    w_prompt: float = 0.05
    w_rel: float = 0.01
    w_onto: float = 0.10
    w_cons: float = 0.20
    w_featcons: float = 0.05

    gamma_pos: float = 1.0
    gamma_neg: float = 2.0

    # Crash-safe checkpoint/logging
    save_every_steps: int = 100
    log_every_steps: int = 50
    keep_step_checkpoint: bool = True

    # Optional debug
    max_train_samples: int = 0
    max_val_samples: int = 0

    def ensure_dirs(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return asdict(self)
