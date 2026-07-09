"""
RTSP 多线程流读取器
- 守护线程持续拉流，read() 永远返回最新帧
- 自动重连：检测到断流后带指数退避重连
"""

import threading
import time

import cv2
from loguru import logger


class RTSPStreamer:
    """多线程 RTSP 读取器，确保 read() 返回最新帧，非阻塞。"""

    def __init__(
        self,
        rtsp_url: str,
        reconnect: bool = True,
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ):
        """
        Args:
            rtsp_url: RTSP 流地址
            reconnect: 是否启用自动重连
            reconnect_delay: 初始重连间隔（秒），每次失败翻倍
            max_reconnect_delay: 最大重连间隔（秒）
        """
        self.rtsp_url = rtsp_url
        self.reconnect_enabled = reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._frame = None
        self._stopped = threading.Event()
        self._connected = False
        self._lock = threading.Lock()

        self._connect()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 内部：连接与读取
    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        """建立 RTSP 连接。返回 True 表示成功。"""
        try:
            self._cap = cv2.VideoCapture(self.rtsp_url)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                self._connected = True
                logger.info("RTSP 连接成功: {}", self.rtsp_url)
                return True
            else:
                logger.warning("RTSP 连接成功但首帧读取失败: {}", self.rtsp_url)
                return False
        except Exception as e:
            logger.error("RTSP 连接异常 ({}): {}", self.rtsp_url, e)
            return False

    def _release(self):
        """释放当前连接。"""
        try:
            if hasattr(self, "_cap") and self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._connected = False

    def _update_loop(self):
        """后台线程：持续读取帧，断线时尝试重连。"""
        delay = self.reconnect_delay

        while not self._stopped.is_set():
            try:
                if not self._connected:
                    # 重连逻辑
                    if self.reconnect_enabled:
                        logger.info(
                            "尝试重连 ({:.1f}s 后)...: {}", delay, self.rtsp_url
                        )
                        if self._stopped.wait(delay):
                            break
                        if self._connect():
                            delay = self.reconnect_delay  # 重置延迟
                            continue
                        else:
                            delay = min(delay * 2, self.max_reconnect_delay)
                            continue
                    else:
                        self._stopped.wait(1)
                        continue

                # 正常读取
                try:
                    ret, frame = self._cap.read()
                    if ret:
                        with self._lock:
                            self._frame = frame
                    else:
                        logger.warning("RTSP 读取失败，断开: {}", self.rtsp_url)
                        self._release()
                        delay = self.reconnect_delay
                except Exception as e:
                    logger.error("RTSP 读取异常: {}", e)
                    self._release()
                    delay = self.reconnect_delay

            except Exception:
                logger.exception(
                    "RTSP 线程未预期异常，1s 后尝试恢复: {}", self.rtsp_url
                )
                self._release()
                delay = self.reconnect_delay
                self._stopped.wait(1)

        self._release()

    # ------------------------------------------------------------------
    # 外部接口
    # ------------------------------------------------------------------
    def read(self):
        """返回最新帧（非阻塞），未连接或尚无帧时返回 None。"""
        with self._lock:
            return self._frame

    @property
    def connected(self) -> bool:
        return self._connected

    def stop(self):
        """停止读取线程并释放资源。"""
        self._stopped.set()
        self._thread.join(timeout=5)
        self._release()
        logger.info("RTSP 流已停止: {}", self.rtsp_url)


class LocalStreamer:
    """本地摄像头读取器（USB / 内置），与 RTSPStreamer 同接口。

    后台线程持续读取，read() 非阻塞返回最新帧。
    """

    def __init__(self, device_id: int = 0):
        """
        Args:
            device_id: OpenCV 摄像头设备 ID，0 = 默认摄像头
        """
        self.device_id = device_id
        self._frame = None
        self._stopped = threading.Event()
        self._connected = False
        self._lock = threading.Lock()

        self._connect()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _connect(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.device_id)
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                self._connected = True
                logger.info("本地摄像头已连接: device_id={}", self.device_id)
                return True
            else:
                logger.warning("本地摄像头打开成功但首帧读取失败: device_id={}", self.device_id)
                return False
        except Exception as e:
            logger.error("本地摄像头连接异常 (device_id={}): {}", self.device_id, e)
            return False

    def _release(self):
        try:
            if hasattr(self, "_cap") and self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._connected = False

    def _update_loop(self):
        while not self._stopped.is_set():
            try:
                if not self._connected:
                    self._stopped.wait(1)
                    continue

                try:
                    ret, frame = self._cap.read()
                    if ret:
                        with self._lock:
                            self._frame = frame
                    else:
                        logger.warning("本地摄像头读取失败: device_id={}", self.device_id)
                        self._release()
                except Exception as e:
                    logger.error("本地摄像头读取异常: {}", e)
                    self._release()

            except Exception:
                logger.exception(
                    "本地摄像头线程未预期异常，1s 后尝试恢复: device_id={}", self.device_id
                )
                self._release()
                self._stopped.wait(1)

        self._release()

    def read(self):
        """返回最新帧（非阻塞）。"""
        with self._lock:
            return self._frame

    @property
    def connected(self) -> bool:
        return self._connected

    def stop(self):
        """停止读取线程并释放摄像头。"""
        self._stopped.set()
        self._thread.join(timeout=5)
        self._release()
        logger.info("本地摄像头已停止: device_id={}", self.device_id)
