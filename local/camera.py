import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    # 1. 加载YOLO模型
    script_dir = Path(__file__).parent.parent
    model_path = script_dir / "runs" / "detect" / "yolo26m_smoking_20260626_0601" / "weights" / "best.pt"

    
    if not model_path.exists():
        print(f"警告：找不到模型文件 {model_path}，将使用普通摄像头模式")
        model = None
    else:
        print(f"加载模型: {model_path}")
        model = YOLO(str(model_path), task="detect")
    
    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("\n按 'q' 键退出")
    print("按 's' 键保存当前帧")
    print("按 'd' 键切换检测模式" if model else "")
    
    detection_mode = True if model else False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法获取画面")
            break
        
        # 3. 如果模型存在且检测模式开启，进行推理
        if model and detection_mode:
            results = model.predict(frame, conf=0.35, verbose=False)
            annotated_frame = results[0].plot()
            
            # 显示检测信息
            detections = len(results[0].boxes) if results[0].boxes else 0
            cv2.putText(annotated_frame, f"Detections: {detections}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            annotated_frame = frame
            if not detection_mode:
                cv2.putText(annotated_frame, "Detection OFF", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 4. 显示画面
        cv2.imshow('Camera with YOLO', annotated_frame)
        
        # 5. 按键处理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('captured_image.jpg', annotated_frame)
            print("图片已保存为 captured_image.jpg")
        elif key == ord('d') and model:
            detection_mode = not detection_mode
            print(f"检测模式: {'ON' if detection_mode else 'OFF'}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()