# yoloCons/experiments/reflection_aug/prototypes/train_vest_cls_aug.py
# 在 manual_cls_aug_split (含反光条离线增强) 上训练分类模型
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26n_vest_cls_aug_{current_time}"

    model = YOLO("yolo26n-cls.pt")

    model.train(
        data=str(PROJECT_ROOT / "dataset" / "manual_cls_aug_split"),
        epochs=300,
        patience=50,
        imgsz=224,
        batch=256,
        device=0,
        name=run_name,
        workers=4,
        project=str(PROJECT_ROOT / "runs"),
    )


if __name__ == "__main__":
    main()
