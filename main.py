import cv2
from ultralytics import YOLO
from pathlib import Path
import os

# COCO 数据集类别名，class 0 = person
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

RTSP_SOURCES = {
    "dahua1": "rtsp://118.140.234.166:8554/dahua1001722",
    "dahua2": "rtsp://118.140.234.166:8554/dahua1000352",
    "dahua3": "rtsp://118.140.130.26:8554/dahua1003362",
}

# 分类标签（根据你的训练数据设置，这里假设二分类）
# 如果你的 vest_cls 数据集标签顺序不同，请调整这里
VEST_CLASS_NAMES = {0: "no_vest", 1: "vest"}


def main():
    script_dir = Path(__file__).parent

    # 1. 加载人体检测模型（默认 YOLO 检测模型）
    detect_model_path = script_dir / "yolo26n.pt"
    if not detect_model_path.exists():
        print(f"错误：找不到检测模型 {detect_model_path}")
        return
    print(f"加载检测模型: {detect_model_path}")
    detect_model = YOLO(str(detect_model_path), task="detect")

    # 2. 加载你训练的二分类模型（vest/no_vest）
    cls_model_path = script_dir / "runs" / "classify" / "yolo26mcls_ppeVest_20260813_1424" / "weights" / "best.pt"
    if not cls_model_path.exists():
        print(f"错误：找不到分类模型 {cls_model_path}")
        return
    print(f"加载分类模型: {cls_model_path}")
    cls_model = YOLO(str(cls_model_path), task="classify")

    # 3. 选择视频源
    # source = 0  # 本地摄像头
    # source = RTSP_SOURCES["dahua1"]  # 使用dahua1
    source = RTSP_SOURCES["dahua2"]  # 使用dahua2
    # source = RTSP_SOURCES["dahua3"]  # 使用dahua3

    print(f"使用视频源: {source}")

    # 4. 设置 RTSP 参数
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)

    if not cap.isOpened():
        print("错误：无法打开视频源")
        return

    ret, frame = cap.read()
    if not ret:
        print("错误：无法获取视频流，请检查RTSP地址是否正确")
        cap.release()
        return

    print("\n按 'q' 键退出")
    print("按 's' 键保存当前帧")
    print("按 'd' 键切换检测模式")

    detection_mode = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法获取画面，尝试重新连接...")
            cap.release()
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if not ret:
                print("重连失败，退出程序")
                break
            continue

        annotated_frame = frame.copy()

        if detection_mode:
            # 第一阶段：检测所有人
            det_results = detect_model.predict(frame, conf=0.35, verbose=False)

            person_count = 0
            vest_count = 0

            boxes = det_results[0].boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    if cls_id != 0:  # 只保留 person 类
                        continue

                    person_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 边界保护
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 裁切人体区域
                    person_crop = frame[y1:y2, x1:x2]

                    # 第二阶段：分类是否为穿背心
                    cls_results = cls_model.predict(person_crop, verbose=False)
                    vest_cls_id = cls_results[0].probs.top1  # 0 或 1
                    vest_conf = cls_results[0].probs.top1conf.item()
                    vest_label = VEST_CLASS_NAMES.get(vest_cls_id, f"cls_{vest_cls_id}")

                    # 根据分类结果设置颜色和统计
                    if vest_cls_id == 1:  # vest
                        color = (0, 255, 0)  # 绿色
                        vest_count += 1
                    else:
                        color = (0, 0, 255)  # 红色

                    # 画框和标签
                    label = f"{vest_label} ({vest_conf:.2f})"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # 顶部状态栏
            cv2.putText(annotated_frame,
                        f"Persons: {person_count} | Vest: {vest_count} | NoVest: {person_count - vest_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "Detection OFF", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Camera with YOLO', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('captured_image.jpg', annotated_frame)
            print("图片已保存为 captured_image.jpg")
        elif key == ord('d'):
            detection_mode = not detection_mode
            print(f"检测模式: {'ON' if detection_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()