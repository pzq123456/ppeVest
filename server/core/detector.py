"""
YOLO 模型封装
- 单例加载，所有摄像头 Worker 共享一个模型实例
- 提供结构化检测结果（ObjectMeta / AttributeMeta）
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

from ultralytics import YOLO
from loguru import logger


# ============================================================================
# 元数据定义（DeepStream 风格：物理实体 + 附属属性分离）
# ============================================================================

@dataclass(frozen=True)
class AttributeMeta:
    """附属属性或细粒度检测结果。

    bbox=None 时为纯分类属性（如 "is_smoking"）；
    bbox 非 None 时为带空间定位的细粒度检测
    （如 cigarette 在人体 ROI 内的精确位置，全帧坐标）。
    """
    name: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ObjectMeta:
    """物理检测实体。

    class_name 与 bbox 严格一致——bbox 就是该物理实体的边界。
    附属的细粒度检测/状态标签存放在 attributes 元组中。

    frozen=True 保证线程安全：跨线程共享时 shallow copy 即足够。
    """
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    attributes: tuple[AttributeMeta, ...] = ()


class SmokeDetector:
    """抽烟检测器，封装 YOLO 模型的加载与推理。

    支持两种模式：
    - 单阶段（默认）：直接在全帧上检测抽烟目标
    - 两阶段（配置 person_model 时启用）：先检测人体，再在人体 ROI 内检测抽烟，显著降低误报
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float | dict[str, float] = 0.35,
        device: int | str | None = 0,
        person_model_path: str | Path | None = None,
        person_conf: float = 0.4,
        imgsz: int | None = None,
    ):
        """
        Args:
            model_path: YOLO 抽烟模型权重路径
            conf: 置信度阈值。float 时全局统一；dict 时逐类设定（如 {'face': 0.5, 'smoking': 0.25}）。
                  YOLO 内部使用 min(dict.values()) 保证不漏检，后置逐类提纯。
            device: GPU 设备 ID（0, 1, ...），None 或 "cpu" 表示 CPU
            person_model_path: 可选的人体检测模型路径（COCO 预训练，class 0=person）。
                               配置后启用两阶段检测：先找人 → 再在人体 ROI 内检测抽烟。
            person_conf: 人体检测置信度阈值（默认 0.4）
            imgsz: 抽烟模型输入尺寸（默认 None，走 YOLO 内部默认值 640）
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 多态 conf：dict 时取最低值给 YOLO 保召回，逐类过滤在 detect() 中完成
        if isinstance(conf, dict):
            self._class_conf = conf
            self._conf = min(conf.values())
        else:
            self._class_conf = None
            self._conf = conf

        logger.info("加载模型: {} (device={}, conf={})", model_path, device,
                    conf if self._class_conf is None else f"{self._conf}(yolo) / {self._class_conf}(per-class)")
        self._model = YOLO(str(model_path))
        self._task = self._model.task  # "detect" 或 "classify"，由模型文件决定
        logger.info("模型任务类型: {}", self._task)

        # 解析 device 参数
        if device is None or device == "cpu":
            self._device = "cpu"
        else:
            self._device = device

        # --- 可选：人体检测模型（第一阶段） ---
        self._person_model = None
        self._person_conf = person_conf
        self._imgsz = imgsz
        if person_model_path is not None:
            person_model_path = Path(person_model_path)
            if not person_model_path.exists():
                raise FileNotFoundError(f"人体检测模型不存在: {person_model_path}")
            logger.info("加载人体检测模型: {} (conf={})", person_model_path, person_conf)
            self._person_model = YOLO(str(person_model_path), task="detect")

    def detect(self, frame) -> list[ObjectMeta]:
        """
        对单帧执行抽烟检测。

        若配置了人体检测模型则走两阶段管线：
        1. 检测人体（COCO class 0）
        2. 在每个人体 ROI 内检测抽烟目标 → 放入 attributes
        未配置人体模型时走单阶段全帧检测。

        Args:
            frame: BGR numpy 数组（来自 cv2）

        Returns:
            ObjectMeta 列表，clas_name 与 bbox 严格一致。
            未检测到任何目标时为空列表。
        """
        if self._person_model is not None:
            return self._detect_two_stage(frame)

        # --- 单阶段模式：全帧检测 ---
        results = self._model.predict(
            frame,
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
            device=self._device,
        )

        objects: list[ObjectMeta] = []
        boxes = results[0].boxes
        if boxes is None:
            return objects

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
            objects.append(ObjectMeta(
                class_name=class_name,
                confidence=confidence,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))

        return objects

    # ------------------------------------------------------------------
    # 两阶段检测（人体 → ROI → 抽烟）
    # ------------------------------------------------------------------
    def _detect_two_stage(self, frame) -> list[ObjectMeta]:
        """
        两阶段检测管线：
        1. 人体检测：用 COCO 预训练模型检测 frame 中所有人（class 0）
        2. ROI 抽烟检测：裁剪每个人体区域，用抽烟模型在该 ROI 内检测
        3. 坐标映射：将 ROI 内抽烟检测坐标映射回全帧坐标，作为 AttributeMeta

        每个人体都会生成一条 ObjectMeta——预览流中所有人体框始终可见。
        告警过滤（仅触发含目标属性的实体）由 AlertManager 按 target_classes 完成。

        Returns:
            ObjectMeta 列表，class_name='person'（两阶段模式下人体为唯一物理实体）。
        """
        h, w = frame.shape[:2]
        objects: list[ObjectMeta] = []

        # --- 第一阶段：人体检测 ---
        person_results = self._person_model.predict(
            frame,
            conf=self._person_conf,
            classes=[0],          # COCO class 0 = person
            verbose=False,
            device=self._device,
        )

        if not person_results or person_results[0].boxes is None:
            return objects

        boxes_data = person_results[0].boxes.data.cpu().numpy()
        if len(boxes_data) == 0:
            return objects

        # --- 第二阶段：逐人体 ROI 检测抽烟 ---
        for box in boxes_data:
            x1, y1, x2, y2 = map(int, box[:4])
            p_conf = float(box[4])  # person 置信度
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            attrs: list[AttributeMeta] = []

            # 仅在 ROI 足够大时执行第二阶段推理
            if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size > 0:
                    if self._task == "classify":
                        # ── 分类模式：对 ROI 做二分类（如 vest / no_vest）──
                        try:
                            cls_results = self._model.predict(
                                person_roi,
                                verbose=False,
                                device=self._device,
                            )
                        except Exception as e:
                            logger.warning(
                                "人体 ROI [{},{} -> {},{}] 分类异常: {}",
                                x1, y1, x2, y2, e,
                            )
                            cls_results = None

                        if cls_results and cls_results[0].probs is not None:
                            top1_id = cls_results[0].probs.top1
                            top1_conf = cls_results[0].probs.top1conf.item()
                            if top1_conf >= self._conf:
                                attr_name = self._model.names.get(top1_id, str(top1_id))
                                attrs.append(AttributeMeta(
                                    name=attr_name,
                                    confidence=top1_conf,
                                    bbox=None,  # 纯分类属性，无子框
                                ))
                    else:
                        # ── 检测模式：在 ROI 内检测细粒度目标（如 cigarette）──
                        try:
                            det_results = self._model.predict(
                                person_roi,
                                conf=self._conf,
                                imgsz=self._imgsz,
                                verbose=False,
                                device=self._device,
                            )
                        except Exception as e:
                            logger.warning(
                                "人体 ROI [{},{} -> {},{}] 检测异常: {}",
                                x1, y1, x2, y2, e,
                            )
                            det_results = None

                        if det_results and det_results[0].boxes is not None:
                            for det_box in det_results[0].boxes:
                                cls_id = int(det_box.cls[0])
                                attr_name = self._model.names.get(cls_id, str(cls_id))
                                attr_conf = float(det_box.conf[0])

                                # 逐类置信度后置过滤
                                if self._class_conf is not None:
                                    min_conf = self._class_conf.get(attr_name, self._conf)
                                    if attr_conf < min_conf:
                                        continue

                                # 子检测坐标映射到全局
                                sx1, sy1, sx2, sy2 = det_box.xyxy[0].tolist()
                                gx1, gy1 = x1 + int(sx1), y1 + int(sy1)
                                gx2, gy2 = x1 + int(sx2), y1 + int(sy2)
                                attrs.append(AttributeMeta(
                                    name=attr_name,
                                    confidence=attr_conf,
                                    bbox=(gx1, gy1, gx2, gy2),
                                ))

            objects.append(ObjectMeta(
                class_name='person',
                confidence=p_conf,
                bbox=(x1, y1, x2, y2),
                attributes=tuple(attrs),
            ))

        return objects

