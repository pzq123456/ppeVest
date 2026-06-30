#!/usr/bin/env python3
"""
帧推送测试接收端
- 接收 binary frame batch (VFb1 协议) → 解析 → 存盘 + 本地 MJPEG 预览
- 用于开发自测，验证协议正确性

用法：
    python local/frame_receiver.py [--port 8080] [--save-dir frames_received]

端点：
    POST /api/vest-frames    — 接收二进制帧批次
    GET  /stream/{cam_id}    — MJPEG 实时预览（浏览器打开）
    GET  /stats              — 接收统计 JSON
"""

import argparse
import asyncio
import struct
import time
import threading
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

# ---------------------------------------------------------------------------
# Binary protocol (VFb1)
# ---------------------------------------------------------------------------
MAGIC = b"VFb1"
MAGIC_LEN = 4
HEADER_LEN = 8   # magic(4) + count(2) + flags(1) + quality(1)
FRAME_META_LEN = 12  # timestamp(8) + data_len(4)


def parse_batch(data: bytes) -> tuple[int, int, list[dict]]:
    """
    解析 VestFrameBatch v1 二进制。

    Returns:
        (frame_count, jpeg_quality, [{"ts_ms": int, "jpeg": bytes}, ...])
    """
    if len(data) < HEADER_LEN:
        raise ValueError(f"数据太短: {len(data)} bytes < {HEADER_LEN}")

    magic, count, flags, quality = struct.unpack_from(">4sHBB", data, 0)
    if magic != MAGIC:
        raise ValueError(f"Magic 不匹配: {magic!r} != {MAGIC!r}")

    frames = []
    offset = HEADER_LEN
    for _ in range(count):
        if offset + FRAME_META_LEN > len(data):
            raise ValueError(f"帧 {len(frames)}/{count} 元数据越界")
        ts, jpeg_len = struct.unpack_from(">qI", data, offset)
        offset += FRAME_META_LEN
        if offset + jpeg_len > len(data):
            raise ValueError(f"帧 {len(frames)}/{count} JPEG 数据越界: {offset}+{jpeg_len} > {len(data)}")
        jpeg = data[offset:offset + jpeg_len]
        offset += jpeg_len
        frames.append({"ts_ms": ts, "jpeg": jpeg})

    return count, quality, frames


# ---------------------------------------------------------------------------
# In-memory frame store
# ---------------------------------------------------------------------------
class FrameStore:
    """线程安全：HTTP handler 写入，stream endpoint 读取。"""

    def __init__(self, save_dir: Path | None = None):
        self._lock = threading.Lock()
        # camera_id → list of (ts_ms, jpeg_bytes)
        self._frames: dict[str, list[tuple[int, bytes]]] = defaultdict(list)
        self._max_frames_per_cam = 100  # MJPEG 回放最多保留的帧
        self._stats: dict[str, dict] = defaultdict(lambda: {"batches": 0, "frames": 0, "last_ts": 0})
        self.save_dir = save_dir

    def add_batch(self, camera_id: str, frames: list[dict]):
        with self._lock:
            cam_frames = self._frames[camera_id]
            for f in frames:
                cam_frames.append((f["ts_ms"], f["jpeg"]))
            # 只保留最近 N 帧
            if len(cam_frames) > self._max_frames_per_cam:
                self._frames[camera_id] = cam_frames[-self._max_frames_per_cam:]

            st = self._stats[camera_id]
            st["batches"] += 1
            st["frames"] += len(frames)
            if frames:
                st["last_ts"] = frames[-1]["ts_ms"]

        # 可选：存盘
        if self.save_dir:
            cam_dir = self.save_dir / camera_id
            cam_dir.mkdir(parents=True, exist_ok=True)
            for f in frames:
                ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(f["ts_ms"] / 1000))
                fname = f"{ts_str}_{f['ts_ms'] % 1000:03d}.jpg"
                (cam_dir / fname).write_bytes(f["jpeg"])

    def get_latest(self, camera_id: str) -> bytes | None:
        with self._lock:
            cam_frames = self._frames.get(camera_id, [])
            return cam_frames[-1][1] if cam_frames else None

    def get_stats(self) -> dict:
        with self._lock:
            return {
                cam: dict(st) for cam, st in self._stats.items()
            }

    @property
    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._frames.keys())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="帧推送测试接收端")
store: FrameStore = None  # type: ignore


@app.on_event("startup")
async def startup():
    global store
    # store 在 main 中创建并注入
    pass


@app.post("/api/vest-frames")
async def receive_frames(request: Request):
    """接收二进制帧批次（VFb1 协议）。"""
    camera_id = request.headers.get("X-Camera-Id", "unknown")
    camera_name = request.headers.get("X-Camera-Name", camera_id)
    batch_seq = request.headers.get("X-Batch-Seq", "?")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="空请求体")

    try:
        count, quality, frames = parse_batch(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"协议解析错误: {e}")

    store.add_batch(camera_id, frames)
    total_size = len(body)
    return JSONResponse({
        "status": "ok",
        "camera_id": camera_id,
        "camera_name": camera_name,
        "batch_seq": batch_seq,
        "frames_received": count,
        "jpeg_quality": quality,
        "batch_size_bytes": total_size,
    })


@app.get("/stream/{camera_id}")
async def stream(camera_id: str):
    """MJPEG 流 — 浏览器打开即可预览最新帧。"""
    async def generate():
        last_ts = 0
        while True:
            # 轮询最新帧
            latest = store.get_latest(camera_id)
            if latest is not None:
                # 只发送新帧
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + latest + b"\r\n")
            await asyncio.sleep(0.05)  # 50ms 轮询 ~20fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stats")
async def stats():
    """JSON 统计。"""
    return JSONResponse({
        "cameras": store.get_stats(),
        "total_cameras": len(store.camera_ids),
    })


@app.get("/")
async def index():
    """简单首页：列出所有摄像头和流链接。"""
    cam_ids = store.camera_ids
    if not cam_ids:
        return HTMLResponse("<h2>等待帧数据...</h2>")

    lines = ["<h2>📹 摄像头列表</h2><ul>"]
    for cid in cam_ids:
        lines.append(
            f'<li><b>{cid}</b> — '
            f'<a href="/stream/{cid}" target="_blank">MJPEG 预览</a>'
            f'</li>'
        )
    lines.append("</ul>")
    lines.append(f'<p><a href="/stats">📊 统计</a></p>')
    return HTMLResponse("\n".join(lines))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="帧推送测试接收端")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="监听端口 (默认 8080)")
    parser.add_argument("--save-dir", default="frames_received", help="帧保存目录")
    args = parser.parse_args()

    global store
    save_dir = Path(args.save_dir)
    store = FrameStore(save_dir=save_dir)

    print(f"🚀 帧推送测试接收端 启动")
    print(f"   监听: http://{args.host}:{args.port}")
    print(f"   接收端点: POST http://{args.host}:{args.port}/api/vest-frames")
    print(f"   实时预览: GET  http://{args.host}:{args.port}/stream/<camera_id>")
    print(f"   统计信息: GET  http://{args.host}:{args.port}/stats")
    if save_dir:
        print(f"   帧存盘: {save_dir.resolve()}/<camera_id>/")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
