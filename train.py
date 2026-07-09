# yoloCons/train.py
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26cls_ppeVest_{current_time}"
    
    model = YOLO("yolo26n-cls.pt")

    model.train(
        data="vest_cls2",
        epochs=300,
        patience=100,
        imgsz=320,
        batch=128,
        device=-1,
        name=run_name,
        workers=4
    )

if __name__ == "__main__":
    main()