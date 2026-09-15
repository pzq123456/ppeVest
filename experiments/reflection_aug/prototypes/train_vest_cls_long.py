# 加长训练：与 train_vest_cls_aug.py 同数据同模型，仅放宽 epoch/patience，
# 让权重更充分稳定（val 为事件级防泄漏划分，预期上限约 0.94，不追求 1.0）。
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26n_vest_cls_long_{current_time}"

    model = YOLO("yolo26n-cls.pt")

    model.train(
        data=str(PROJECT_ROOT / "dataset" / "manual_cls_aug_split"),
        epochs=500,
        patience=150,
        imgsz=224,
        batch=256,
        device=0,
        name=run_name,
        workers=4,
        project=str(PROJECT_ROOT / "runs"),
    )


if __name__ == "__main__":
    main()
