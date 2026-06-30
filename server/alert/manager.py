"""
告警管理器
- 冷却期控制：同一摄像头两次告警的最小间隔
- 连续帧确认：连续 N 帧检测到才触发告警（防止单帧误报）
- 帧标注：检测框 + 可选文字水印 → base64 编码
- 告警分发：调用 Webhook 等渠道推送
"""

import base64
import time
from datetime import datetime, timezone

import cv2
from loguru import logger

from server.core.detector import Detection
from server.alert.webhook import WebhookAlerter


class AlertManager:
    """管理单路摄像头的告警状态与分发。"""

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        target_classes: list[str] | None = None,
        save_frame_overlay: bool = False,
        cooldown_seconds: float = 30.0,
        min_detection_count: int = 3,
        webhook: WebhookAlerter | None = None,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头显示名称
            target_classes: 触发告警的目标类别列表，None 则匹配所有检测
            save_frame_overlay: 是否在证据帧上叠加摄像头名称/时间水印
            cooldown_seconds: 同一摄像头两次告警的最小间隔（秒）
            min_detection_count: 连续检测到抽烟的帧数阈值
            webhook: Webhook 推送实例，None 则不推送
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.target_classes = target_classes
        self.save_frame_overlay = save_frame_overlay
        self.cooldown_seconds = cooldown_seconds
        self.min_detection_count = min_detection_count
        self.webhook = webhook

        # 内部状态
        self._consecutive_hits = 0          # 连续检测到抽烟的帧数
        self._last_alert_time: float = 0.0  # 上一次告警的 Unix 时间戳

    # ------------------------------------------------------------------
    # 主入口：每帧调用
    # ------------------------------------------------------------------
    def handle(self, frame, detections: list[Detection]) -> bool:
        """
        处理一帧的检测结果，决定是否触发告警。

        Args:
            frame: 原始帧（未标注）
            detections: detect() 返回的检测列表

        Returns:
            True 表示触发了告警
        """
        try:
            # 按配置的目标类别过滤（None 表示匹配所有检测）
            if self.target_classes is not None:
                smoking_detections = [
                    d for d in detections if d.class_name in self.target_classes
                ]
            else:
                smoking_detections = detections

            if smoking_detections:
                self._consecutive_hits += 1
                logger.debug(
                    "[{}] 检测到目标: {} 个, 连续 {}/{} 帧",
                    self.camera_name,
                    len(smoking_detections),
                    self._consecutive_hits,
                    self.min_detection_count,
                )

                if self._consecutive_hits >= self.min_detection_count:
                    if self._cooldown_passed():
                        self._trigger(frame, smoking_detections)
                        self._consecutive_hits = 0
                        return True
                    else:
                        # 冷却中，重置计数器避免冷却结束后立即再次触发
                        self._consecutive_hits = 0
            else:
                # 无检测 → 递减计数器（逐步递减，容忍偶尔丢帧）
                if self._consecutive_hits > 0:
                    self._consecutive_hits -= 1

            return False
        except Exception:
            logger.exception("[{}] 告警处理异常，跳过本帧", self.camera_name)
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _cooldown_passed(self) -> bool:
        """检查冷却期是否已过。"""
        elapsed = time.time() - self._last_alert_time
        return elapsed >= self.cooldown_seconds

    def _trigger(self, frame, detections: list[Detection]):
        """触发告警：标注帧 → base64 编码 → 推送 Webhook。"""
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()

        logger.info(
            "🚨 告警触发 [{}] {} 个目标",
            self.camera_name, len(detections),
        )

        # 1. 标注检测框
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(
                annotated, label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        # 2. 可选：叠加摄像头/时间水印
        if self.save_frame_overlay:
            overlay_text = [
                f"Camera: {self.camera_name} ({self.camera_id})",
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            for i, text in enumerate(overlay_text):
                cv2.putText(
                    annotated, text,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

        # 3. 编码为 base64
        _, buffer = cv2.imencode(".jpg", annotated)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # 4. 构建告警 payload
        payload = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "timestamp": iso_timestamp,
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "bbox": list(d.bbox),
                }
                for d in detections
            ],
            "frame_base64": frame_base64,
        }

        # 5. 推送 Webhook（失败不影响主流程）
        if self.webhook:
            try:
                self.webhook.send(payload)
            except Exception:
                logger.exception("Webhook 推送异常")

        self._last_alert_time = time.time()
