from pathlib import Path
from ultralytics import YOLO

script_dir = Path(__file__).parent.parent

model_path = script_dir / "runs" / "classify" / "yolo26cls_ppeVest_20260709_1036" / "weights" / "best.pt"
data_path = script_dir / "vest_cls2" / "data.yaml"

if __name__ == "__main__":
    model = YOLO(model_path)

    model.export(format="onnx", data=data_path)