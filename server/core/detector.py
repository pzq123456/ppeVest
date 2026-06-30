"""
YOLO 模型封装
- 支持两种模式：single（单模型检测）和 cascade（检测+分类级联）
- 单例加载，所有摄像头 Worker 共享一个模型实例
- 提供结构化检测结果
"""

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """单条检测结果。"""
    class_name: str        # 类别名称，如 "vest" / "no_vest"
    confidence: float      # 置信度 0-1
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)


class SmokeDetector:
    """
    检测器，封装 YOLO 模型的加载与推理。

    mode = "single": 单 YOLO 检测模型（原抽烟检测逻辑）
    mode = "cascade": 级联模型 — Stage1 检测 person → Stage2 分类 vest/no_vest
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self, model_cfg: dict):
        """
        Args:
            model_cfg: config.yaml 中 model 节的完整字典
        """
        self._mode = model_cfg.get("type", "single")
        self._device = _resolve_device(model_cfg.get("device", 0))

        if self._mode == "cascade":
            self._init_cascade(model_cfg)
        else:
            self._init_single(model_cfg)

    # ------------------------------------------------------------------
    # single 模式初始化
    # ------------------------------------------------------------------
    def _init_single(self, cfg: dict):
        model_path = cfg["path"]
        conf = cfg.get("conf", 0.35)

        # 多态 conf：dict 时取最低值给 YOLO 保召回，逐类过滤在 detect() 中完成
        if isinstance(conf, dict):
            self._class_conf = conf
            self._conf = min(conf.values())
        else:
            self._class_conf = None
            self._conf = conf

        logger.info("加载单模型: {} (device={}, conf={})", model_path, self._device,
                    conf if self._class_conf is None else f"{self._conf}(yolo) / {self._class_conf}(per-class)")
        self._model = self._load_model(model_path, task="detect")

    # ------------------------------------------------------------------
    # cascade 模式初始化
    # ------------------------------------------------------------------
    def _init_cascade(self, cfg: dict):
        detect_cfg = cfg["detect"]
        classify_cfg = cfg["classify"]

        self._detect_conf = detect_cfg.get("conf", 0.35)
        self._target_class_id = detect_cfg.get("target_class_id", 0)
        self._cls_conf = classify_cfg.get("conf", 0.5)
        raw_labels = classify_cfg.get("labels", {0: "no_vest", 1: "vest"})
        # 兼容 YAML 列表格式 ["no_vest", "vest"] 和字典格式 {0: "no_vest", 1: "vest"}
        if isinstance(raw_labels, list):
            self._cls_labels = {i: name for i, name in enumerate(raw_labels)}
        else:
            self._cls_labels = raw_labels

        # 加载检测模型（自动下载：纯文件名如 yolo11n.pt → ultralytics 自动拉取）
        logger.info("加载检测模型: {} (device={}, conf={})",
                    detect_cfg["model"], self._device, self._detect_conf)
        self._model = self._load_model(detect_cfg["model"], task="detect")

        # 加载分类模型
        logger.info("加载分类模型: {} (device={}, conf={})",
                    classify_cfg["path"], self._device, self._cls_conf)
        self._cls_model = self._load_model(classify_cfg["path"], task="classify")

        # cascade 模式不使用逐类 conf
        self._class_conf = None
        self._conf = self._detect_conf

    # ------------------------------------------------------------------
    # 模型加载（支持自动下载）
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(model_path: str, task: str) -> YOLO:
        """
        加载 YOLO 模型。
        - 纯文件名（如 yolo11n.pt）→ ultralytics 自动下载到工作目录
        - 相对/绝对路径 → 从本地加载
        """
        path = Path(model_path)
        # 如果包含路径分隔符，检查文件是否存在
        if path.parent != Path(".") and not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        try:
            model = YOLO(str(path) if path.parent != Path(".") else model_path,
                         task=task)
        except Exception as e:
            if path.parent == Path("."):
                logger.error("模型加载失败（自动下载也可能失败）: {} — {}", model_path, e)
            raise

        logger.info("模型已就绪: {}", model_path)
        return model

    # ------------------------------------------------------------------
    # 推理入口
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        对单帧执行检测。

        Args:
            frame: BGR numpy 数组（来自 cv2）

        Returns:
            Detection 列表
        """
        if self._mode == "cascade":
            return self._detect_cascade(frame)
        return self._detect_single(frame)

    # ------------------------------------------------------------------
    # single 模式推理（原逻辑）
    # ------------------------------------------------------------------
    def _detect_single(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self._conf, verbose=False, device=self._device,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self._model.names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])

            # 逐类置信度后置过滤
            if self._class_conf is not None:
                min_conf = self._class_conf.get(class_name, self._conf)
                if confidence < min_conf:
                    continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))

        return detections

    # ------------------------------------------------------------------
    # cascade 模式推理
    # ------------------------------------------------------------------
    def _detect_cascade(self, frame: np.ndarray) -> list[Detection]:
        """
        Stage 1: 检测所有人（person class）
        Stage 2: 对每个人体区域做 vest/no_vest 分类
        """
        # Stage 1: 检测 person
        det_results = self._model.predict(
            frame, conf=self._detect_conf, verbose=False, device=self._device,
        )

        detections: list[Detection] = []
        boxes = det_results[0].boxes
        if boxes is None:
            return detections

        h, w = frame.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != self._target_class_id:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Stage 2: 裁切人体区域 → 分类
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            cls_results = self._cls_model.predict(
                person_crop, verbose=False, device=self._device,
            )

            if cls_results[0].probs is None:
                continue

            cls_id = cls_results[0].probs.top1
            cls_conf = cls_results[0].probs.top1conf.item()

            # 分类置信度过滤
            if cls_conf < self._cls_conf:
                continue

            label = self._cls_labels[cls_id] if cls_id in self._cls_labels else f"cls_{cls_id}"

            detections.append(Detection(
                class_name=label,
                confidence=round(cls_conf, 3),
                bbox=(x1, y1, x2, y2),
            ))

        return detections

    # ------------------------------------------------------------------
    # 帧标注
    # ------------------------------------------------------------------
    def annotate_frame(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        在帧上绘制检测框和标签（不修改原图，返回新图）。

        Args:
            frame: 原始帧
            detections: detect() 返回的检测列表

        Returns:
            标注后的帧
        """
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            # 根据类别选颜色
            if d.class_name == "vest":
                color = (0, 255, 0)   # 绿色 = 穿了反光衣
            elif d.class_name == "no_vest":
                color = (0, 0, 255)   # 红色 = 未穿
            else:
                color = (0, 0, 255)   # 默认红色

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return annotated


# ============================================================================
# 工具函数
# ============================================================================
def _resolve_device(device) -> str | int:
    """解析 device 参数。"""
    if device is None or device == "cpu":
        return "cpu"
    return device
