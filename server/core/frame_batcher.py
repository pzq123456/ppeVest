"""
帧批量收集器
- 每个摄像头一个实例
- 线程安全：Worker 线程写入，推送线程读取
- 按时间窗口 + 帧数上限自动打包
"""

import time
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger


@dataclass
class FramePacket:
    """单个待发送帧（JPEG 已编码）。"""
    timestamp_ms: int
    jpeg_bytes: bytes


class FrameBatcher:
    """
    收集标注帧，到时间窗口后吐出批次包。

    用法（在 CameraWorker._run 中）:
        seq = self._batcher.add(frame, annotated)
        if seq is not None:
            seq, batch = self._batcher.drain()
            self._pusher.send(..., seq, batch, ...)
    """

    def __init__(
        self,
        camera_id: str,
        batch_interval_ms: int = 1000,
        max_frames: int = 15,
        jpeg_quality: int = 70,
    ):
        """
        Args:
            camera_id: 摄像头标识（仅用于日志）
            batch_interval_ms: 批次间隔（毫秒）
            max_frames: 单批次最多保留多少帧（超限时丢弃旧帧）
            jpeg_quality: JPEG 编码质量 1-100
        """
        self.camera_id = camera_id
        self.batch_interval_s = batch_interval_ms / 1000.0
        self.max_frames = max_frames
        self.jpeg_quality = jpeg_quality

        self._buffer: list[FramePacket] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._seq = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def add(self, annotated_bgr: np.ndarray) -> int | None:
        """
        Worker 每帧调用，编码并加入缓冲区。

        Args:
            annotated_bgr: 已标注的 BGR 帧

        Returns:
            None — 未到发送时间，帧已缓存
            int  — 批次序号，表示该发送了（调用方应立即 drain）
        """
        # JPEG 编码（在 Worker 线程做，不在推送线程做）
        ok, jpg = cv2.imencode(
            ".jpg", annotated_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            logger.warning("[{}] JPEG 编码失败，跳过本帧", self.camera_id)
            return None

        pkt = FramePacket(
            timestamp_ms=int(time.time() * 1000),
            jpeg_bytes=jpg.tobytes(),
        )

        with self._lock:
            self._buffer.append(pkt)
            # 超限裁剪：保留最新 N 帧
            if len(self._buffer) > self.max_frames:
                self._buffer = self._buffer[-self.max_frames:]

        # 时间窗口检查
        now = time.time()
        if now - self._last_flush >= self.batch_interval_s:
            self._last_flush = now
            self._seq += 1
            return self._seq
        return None

    def drain(self) -> tuple[int, list[FramePacket]]:
        """
        取出当前批次所有帧（清空缓冲区）。

        Returns:
            (seq, packets) — seq=0 表示无数据
        """
        with self._lock:
            batch = self._buffer.copy()
            self._buffer.clear()

        if not batch:
            return 0, []

        return self._seq, batch
