# yoloCons/scripts/train_vest_cls.py
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26n_vest_cls_{current_time}"

    model = YOLO("yolo26n-cls.pt")

    model.train(
        data="vest_cls",
        epochs=300,
        patience=20,
        imgsz=224,
        batch=256,
        device=0,
        name=run_name,
        workers=4,
        project=str(PROJECT_ROOT / "runs"),
    )


if __name__ == "__main__":
    main()
