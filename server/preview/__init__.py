"""视频流预览模块 — MJPEG over HTTP.

Worker 能力已合并至 server.core.camera_worker.CameraWorker（jpeg_quality 参数）。
本模块仅保留 FastAPI 路由 + lifespan。
"""
