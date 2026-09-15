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
        data="dataset/manual_cls",
        epochs=1000,
        patience=150,  # 从100增加到150
        imgsz=320,
        batch=32,      # ⭐调整batch
        device=-1,
        name=run_name,
        workers=4,
        amp=False,
        
        # 1. 数据增强 ⭐
        hsv_h=0.02,
        hsv_s=0.8,     # ⭐最重要
        hsv_v=0.5,     # ⭐最重要
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        
        # 2. 学习率调度 ⭐
        cos_lr=True,
        lrf=0.001,
                
        # 4. 优化器 ⭐
        optimizer='AdamW',
        weight_decay=0.0005,
        
        # 5. 预热
        warmup_epochs=3,
        warmup_momentum=0.8,
    )

if __name__ == "__main__":
    main()