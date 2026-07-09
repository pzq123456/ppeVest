"""
单路摄像头 Worker — 三管线解耦架构
- Pipeline 1（帧捕获）: Streamer 后台线程持续拉流，read() 非阻塞返回最新帧
- Pipeline 2（检测管线）: 独立线程 — 读帧 → YOLO 推理 → 更新共享 ObjectMeta → 告警决策
- Pipeline 3（预览管线）: 独立线程 — 以固定帧率读帧 → render_osd() → JPEG → asyncio.Queue
                          同时缓存渲染帧作为告警证据帧快照（单一渲染源）

检测框随检测结果实时更新，预览帧率不受 YOLO 推理速度 / Webhook 阻塞影响。
告警证据帧直接复用预览管线的渲染结果，避免二次渲染和 GIL 竞争。
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import simplejpeg
from loguru import logger

from server.core.detector import SmokeDetector, ObjectMeta
from server.alert.manager import AlertManager
from server.utils.draw import render_osd


class CameraWorker:
    """负责一路摄像头的检测与告警，可选输出 MJPEG 预览帧。

    三管线全部运行在独立线程中，通过线程安全的共享状态交换数据：
    - Streamer 线程（外部）→ 最新原始帧（Lock 保护）
    - 检测线程 → 最新 ObjectMeta 列表（_latest_objects + Lock）
    - 预览线程 → render_osd() 标注后的 JPEG 帧（asyncio.Queue 桥接至 HTTP 协程）
                + 缓存渲染帧副本（_latest_rendered + Lock）供告警证据复用

    告警路径仅做决策，通过 snapshot() 获取预览管线已渲染的帧。
    JPEG 编码在 executor 线程，base64/HTTP 推送由 daemon 线程 fire-and-forget。
    """

    DEFAULT_PREVIEW_FPS: float = 24

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        streamer,
        detector: SmokeDetector,
        alert_manager: AlertManager,
        jpeg_quality: int | None = None,
        status_interval: int = 100,
        summary_interval: float = 60.0,
        preview_fps: float = DEFAULT_PREVIEW_FPS,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头名称（用于日志和告警）
            streamer: 视频流读取器
            detector: 共享的 SmokeDetector 实例
            alert_manager: 该摄像头的 AlertManager 实例
            jpeg_quality: JPEG 质量 1-100，None 表示不编码（headless 模式）
            status_interval: 每隔多少帧打印一次 DEBUG 状态日志（默认 100）
            summary_interval: 每隔多少秒打印一次 INFO 摘要（默认 60）
            preview_fps: 预览渲染的目标帧率（默认 15）
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._streamer = streamer
        self.detector = detector
        self.alert_manager = alert_manager
        self._jpeg_quality = jpeg_quality
        self.status_interval = status_interval
        self.summary_interval = summary_interval
        self._preview_fps = preview_fps

        self._stopped = threading.Event()

        # ── 共享检测结果（检测线程写入 → 预览线程读取）──
        # ObjectMeta 为 frozen=True，shallow copy 即可线程安全共享
        self._latest_objects: list[ObjectMeta] = []
        self._objects_lock = threading.Lock()

        # ── 共享渲染帧（预览线程写入 → 告警证据读取）──
        # 预览线程每次 render_osd() 后缓存，alert 路径通过 snapshot() 获取
        self._latest_rendered = None  # np.ndarray | None
        self._rendered_lock = threading.Lock()

        # ── 预览 Pub/Sub（仅 jpeg_quality 非 None 时启用）──
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fps: float = 0.0
        self._preview_queue: asyncio.Queue | None = (
            asyncio.Queue(maxsize=1) if jpeg_quality is not None else None
        )

        # ── 线程 ──
        self._detect_thread: threading.Thread | None = None
        self._preview_thread: threading.Thread | None = None

        # ── Webhook 异步执行器（避免告警推送阻塞检测/预览管线）──
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"webhook-{camera_id}",
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """启动检测线程和预览线程。"""
        self._detect_thread = threading.Thread(
            target=self._detect_loop, daemon=True,
            name=f"detect-{self.camera_id}",
        )
        self._detect_thread.start()

        if self._jpeg_quality is not None:
            self._preview_thread = threading.Thread(
                target=self._preview_loop, daemon=True,
                name=f"preview-{self.camera_id}",
            )
            self._preview_thread.start()

        logger.debug("[{}] Worker 线程已启动 (detect{})",
                     self.camera_name,
                     " + preview" if self._jpeg_quality is not None else "")

    def stop(self):
        """停止所有线程并释放资源。"""
        self._stopped.set()

        # 向预览队列发送终止哨兵，唤醒正在等待的 MJPEG 协程
        preview_queue = self._preview_queue
        loop = self._loop
        if preview_queue is not None and loop is not None:
            try:
                async def _send_sentinel(q: asyncio.Queue):
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await q.put(None)
                asyncio.run_coroutine_threadsafe(_send_sentinel(preview_queue), loop)
            except Exception:
                pass

        if self._streamer:
            self._streamer.stop()

        for t in (self._detect_thread, self._preview_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5)

        self._executor.shutdown(wait=False)
        logger.debug("[{}] Worker 已停止", self.camera_name)

    @property
    def is_running(self) -> bool:
        return self._detect_thread is not None and self._detect_thread.is_alive()

    # ------------------------------------------------------------------
    # 公共：获取渲染帧快照（告警证据用）
    # ------------------------------------------------------------------
    def snapshot(self):
        """返回最新 OSD 渲染帧的线程安全副本。

        预览管线每次 render_osd() 后更新 _latest_rendered，
        本方法在调用线程上执行 copy，锁仅保护引用读取。
        返回 None 表示尚无渲染帧可用。
        """
        with self._rendered_lock:
            frame = self._latest_rendered
        if frame is None:
            return None
        return frame.copy()

    # ==================================================================
    # Pipeline 2: 检测线程
    # ==================================================================
    def _detect_loop(self):
        """检测主循环。独立于预览管线运行。

        1. 从 Streamer 读取最新帧（非阻塞）
        2. YOLO 推理 → List[ObjectMeta]
        3. Lock 保护下写入 _latest_objects（预览线程读取）
        4. 告警处理（Webhook 在独立线程池中执行，不阻塞本循环）
        """
        frame_count = 0
        alert_count = 0
        t_start = time.time()
        t_last_summary = t_start

        logger.info("[{}] 检测管线已启动", self.camera_name)

        while not self._stopped.is_set():
            try:
                # 1. 读取帧
                t1 = time.time()
                frame = self._streamer.read()
                if frame is None:
                    if not self._streamer.connected:
                        self._stopped.wait(1)
                    continue

                # 2. YOLO 推理 → List[ObjectMeta]
                try:
                    objects = self.detector.detect(frame)
                except Exception as e:
                    logger.error("[{}] 检测异常: {}", self.camera_name, e)
                    continue

                t2 = time.time()

                # 3. 发布共享检测结果（预览线程读取，shallow copy 安全）
                with self._objects_lock:
                    self._latest_objects = objects

                # 4. 告警处理 — 从预览管线获取已渲染帧快照，Webhook 全链路异步
                try:
                    snapshot = self.snapshot()
                    if self.alert_manager.handle(objects, snapshot, executor=self._executor):
                        alert_count += 1
                except Exception:
                    logger.exception("[{}] 告警处理异常", self.camera_name)

                # 5. 统计
                frame_count += 1
                now = time.time()
                now_elapsed = now - t_start

                if frame_count % self.status_interval == 0 or now - t_last_summary >= self.summary_interval:
                    self._fps = frame_count / now_elapsed if now_elapsed > 0 else 0

                if frame_count % self.status_interval == 0:
                    inference_ms = (t2 - t1) * 1000
                    logger.debug(
                        "[{}] 帧: {} | 推理: {:.1f}ms | FPS: {:.1f} | 运行: {:.0f}s",
                        self.camera_name, frame_count, inference_ms, self._fps, now_elapsed,
                    )

                if now - t_last_summary >= self.summary_interval:
                    logger.info("status camera={} fps={:.1f} frames={} alerts={} uptime={:.0f}s",
                                self.camera_name, self._fps, frame_count, alert_count, now_elapsed)
                    t_last_summary = now

            except Exception:
                logger.exception(
                    "[{}] 检测线程未预期异常，跳过本帧继续运行", self.camera_name
                )
                time.sleep(0.1)

        elapsed = time.time() - t_start
        logger.info("[{}] 检测管线停止 frames={} alerts={} uptime={:.0f}s fps={:.1f}",
                    self.camera_name, frame_count, alert_count, elapsed,
                    frame_count / elapsed if elapsed > 0 else 0)

    # ==================================================================
    # Pipeline 3: 预览渲染线程
    # ==================================================================
    def _preview_loop(self):
        """预览渲染循环。完全独立于检测管线。

        1. 以固定帧率从 Streamer 读取最新帧
        2. Lock 下读取 _latest_objects（shallow copy，frozen 元数据安全）
        3. render_osd(style="preview") 工业风标注
        4. JPEG 编码 → asyncio.Queue 发布
        """
        frame_interval = 1.0 / self._preview_fps
        frame_count = 0

        logger.info("[{}] 预览管线已启动 target_fps={:.0f}",
                     self.camera_name, self._preview_fps)

        while not self._stopped.is_set():
            t_start = time.time()

            try:
                # 1. 读取最新帧
                frame = self._streamer.read()
                if frame is None:
                    self._stopped.wait(0.01)
                    continue

                # 2. 读取最新检测元数据（Lock + shallow copy）
                with self._objects_lock:
                    objects = list(self._latest_objects)

                # 2.5 仅渲染匹配 target_classes 属性的对象（如只画 no_vest）
                target_classes = self.alert_manager.target_classes
                if target_classes:
                    objects = [obj for obj in objects
                               if any(attr.name in target_classes for attr in obj.attributes)]

                # 3. 工业风 OSD 渲染（纯函数，不修改入参）— 单一渲染源
                rendered = render_osd(frame, objects, style="preview")

                # 4. 缓存为告警证据帧快照（单次渲染，预览 + 告警双用途）
                with self._rendered_lock:
                    self._latest_rendered = rendered

                # 5. JPEG 编码
                jpeg_q = self._jpeg_quality
                assert jpeg_q is not None, "_preview_loop 仅在 preview 模式下运行"
                buf = simplejpeg.encode_jpeg(
                    rendered, quality=jpeg_q, colorspace='BGR')

                # 6. 发布到 asyncio 队列
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self._enqueue_preview(buf), self._loop
                    )

                frame_count += 1

            except Exception:
                logger.exception("[{}] 预览线程异常，跳过本帧", self.camera_name)

            # 维持目标帧率
            elapsed = time.time() - t_start
            if elapsed < frame_interval:
                self._stopped.wait(frame_interval - elapsed)

        logger.info("[{}] 预览管线停止 frames={}", self.camera_name, frame_count)

    async def _enqueue_preview(self, buf: bytes):
        """在事件循环中执行：清空队列并放入最新帧。

        Queue(maxsize=1) + get_nowait() 保证队列中始终只有最新一帧。
        State 3（QueueFull）静默丢弃：同一 66ms 窗口内已有足够新鲜的帧。
        """
        q = self._preview_queue
        assert q is not None, "_enqueue_preview 仅在 preview 模式下调用"
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(buf)
        except asyncio.QueueFull:
            pass
