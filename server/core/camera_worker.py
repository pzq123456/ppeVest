"""
单路摄像头 Worker
- 每个摄像头一个独立线程
- 主循环：读取帧 → 检测 → 告警处理
- 定期输出状态日志
"""

import time
import threading

from loguru import logger

from server.core.detector import SmokeDetector, Detection
from server.alert.manager import AlertManager


class CameraWorker:
    """负责一路摄像头的检测与告警。"""

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        streamer,           # RTSPStreamer | LocalStreamer — 任何有 read()/stop()/connected 的对象
        detector: SmokeDetector,
        alert_manager: AlertManager,
        status_interval: int = 100,
        summary_interval: float = 60.0,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头名称（用于日志和告警）
            streamer: 视频流读取器（RTSPStreamer / LocalStreamer），由外部创建
            detector: 共享的 SmokeDetector 实例
            alert_manager: 该摄像头的 AlertManager 实例
            status_interval: 每隔多少帧打印一次 DEBUG 状态日志（默认 100）
            summary_interval: 每隔多少秒打印一次 INFO 摘要（默认 60）
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._streamer = streamer
        self.detector = detector
        self.alert_manager = alert_manager
        self.status_interval = status_interval
        self.summary_interval = summary_interval

        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """启动 Worker 线程。"""
        # streamer 已由外部创建并传入，直接使用
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"worker-{self.camera_id}"
        )
        self._thread.start()
        logger.info("[{}] Worker 已启动", self.camera_name)

    def stop(self):
        """停止 Worker 线程并释放资源。"""
        self._stopped.set()
        if self._streamer:
            self._streamer.stop()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[{}] Worker 已停止", self.camera_name)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _run(self):
        """Worker 主循环。"""
        frame_count = 0
        t_start = time.time()
        t_last_summary = t_start

        logger.info("[{}] 开始检测", self.camera_name)

        while not self._stopped.is_set():
            try:
                # 1. 读取帧
                t1 = time.time()
                frame = self._streamer.read()
                if frame is None:
                    if not self._streamer.connected:
                        self._stopped.wait(1)
                    continue

                # 2. 检测
                try:
                    detections = self.detector.detect(frame)
                except Exception as e:
                    logger.error("[{}] 检测异常: {}", self.camera_name, e)
                    continue

                t2 = time.time()
                now = time.time()

                # 3. 告警处理（防御：handle 内部异常不杀死 Worker）
                if detections:
                    try:
                        self.alert_manager.handle(frame, detections)
                    except Exception:
                        logger.exception("[{}] 告警处理异常", self.camera_name)

                # 4. 统计
                frame_count += 1

                # -- DEBUG: 逐帧统计（仅 DEBUG 级别可见） --
                if frame_count % self.status_interval == 0:
                    elapsed = now - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    inference_ms = (t2 - t1) * 1000
                    logger.debug(
                        "[{}] 帧: {} | 推理: {:.1f}ms | FPS: {:.1f} | 运行: {:.0f}s",
                        self.camera_name, frame_count, inference_ms, fps, elapsed,
                    )

                # -- INFO: 时间摘要（默认每 60 秒一条） --
                if now - t_last_summary >= self.summary_interval:
                    elapsed = now - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        "[{}] 运行 {}s | 已处理 {} 帧 | FPS {:.1f}",
                        self.camera_name, int(elapsed), frame_count, fps,
                    )
                    t_last_summary = now

            except Exception:
                logger.exception(
                    "[{}] Worker 线程未预期异常，跳过本帧继续运行", self.camera_name
                )
                time.sleep(0.1)

        # 退出统计
        elapsed = time.time() - t_start
        logger.info(
            "[{}] 停止 — 总帧数: {}, 运行: {:.0f}s, 平均 FPS: {:.1f}",
            self.camera_name, frame_count, elapsed,
            frame_count / elapsed if elapsed > 0 else 0,
        )
