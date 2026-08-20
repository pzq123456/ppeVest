# yoloCons/train.py
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26mcls_ppeVest_{current_time}"
    
    model = YOLO("yolo26m-cls.pt")

    model.train(
        data="dataset/NoSuit_cls_clean_aug",
        epochs=500,
        patience=100,
        imgsz=320,
        batch=32,
        device=-1,
        name=run_name,
        workers=4,
        amp=False,
    )

if __name__ == "__main__":
    main()