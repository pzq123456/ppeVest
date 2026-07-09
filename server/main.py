#!/usr/bin/env python3
"""
抽烟检测服务端入口

启动方式：
    python -m server.main                  # 使用默认配置 server/config.yaml
    python -m server.main --config my.yaml # 指定配置文件

工作流程：
    1. 加载并校验 YAML 配置
    2. 初始化日志
    3. 加载 YOLO 模型（所有摄像头共享）
    4. 为每个 enabled 的摄像头创建 CameraWorker（可输出 MJPEG 预览帧）
    5. 可选：启动 MJPEG 预览 HTTP 服务（preview 节存在时启用）
    6. 等待退出信号，优雅关闭
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中（支持 python -m server.main 和 python server/main.py）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from server.utils.logger import setup_logger
from server.core.detector import SmokeDetector
from server.core.streamer import RTSPStreamer, LocalStreamer
from server.core.camera_worker import CameraWorker  # 含可选 JPEG 输出能力
from server.alert.webhook import WebhookAlerter
from server.alert.manager import AlertManager


# ============================================================================
# 配置加载
# ============================================================================
def load_config(config_path: str) -> dict:
    """加载并校验 YAML 配置文件。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 必填字段校验
    _validate_config(config, path)
    return config


def _validate_config(config: dict, path: Path):
    """简单的配置校验。"""
    errors = []

    if "model" not in config:
        errors.append("缺少 'model' 节")
    else:
        if "path" not in config["model"]:
            errors.append("model.path 为必填项")

    if "cameras" not in config:
        errors.append("缺少 'cameras' 节")
    else:
        cameras = config["cameras"]
        if not isinstance(cameras, list) or len(cameras) == 0:
            errors.append("cameras 必须是非空列表")
        else:
            for i, cam in enumerate(cameras):
                for key in ("id", "name"):
                    if key not in cam:
                        errors.append(f"cameras[{i}].{key} 为必填项")
                cam_type = cam.get("type", "rtsp")
                if cam_type == "rtsp" and "rtsp_url" not in cam:
                    errors.append(f"cameras[{i}].rtsp_url 为必填项 (type=rtsp)")
                if cam_type not in ("rtsp", "local"):
                    errors.append(f"cameras[{i}].type 无效: {cam_type}，支持 rtsp / local")

    if "alert" not in config:
        errors.append("缺少 'alert' 节")

    # preview 节校验（可选，仅在存在时检查）
    if "preview" in config:
        preview = config["preview"]
        if not isinstance(preview, dict):
            errors.append("preview 必须是字典")
        else:
            port = preview.get("port")
            if port is not None and not isinstance(port, int):
                errors.append("preview.port 必须为整数")
            jpeg_quality = preview.get("jpeg_quality")
            if jpeg_quality is not None and not (isinstance(jpeg_quality, int) and 1 <= jpeg_quality <= 100):
                errors.append("preview.jpeg_quality 必须为 1-100 的整数")

    if errors:
        msg = f"配置校验失败 ({path}):\n  " + "\n  ".join(errors)
        raise ValueError(msg)


# ============================================================================
# Streamer 工厂
# ============================================================================
def _create_streamer(cam: dict):
    """根据摄像头配置创建对应的视频流读取器。"""
    cam_type = cam.get("type", "rtsp")
    if cam_type == "local":
        return LocalStreamer(device_id=cam.get("device_id", 0))
    else:
        return RTSPStreamer(rtsp_url=cam["rtsp_url"])


def _start_preview_server(workers_dict: dict, preview_cfg: dict):
    """启动预览 HTTP 服务器（daemon 线程）。"""
    import threading
    import uvicorn
    from server.preview.app import create_app

    preview_app = create_app(workers_dict)
    threading.Thread(
        target=lambda: uvicorn.run(
            preview_app,
            host=preview_cfg.get("host", "0.0.0.0"),
            port=preview_cfg.get("port", 8080),
            log_level="warning",
        ),
        daemon=True,
    ).start()


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="抽烟检测服务端")
    parser.add_argument(
        "--config", "-c",
        default=str(_PROJECT_ROOT / "server" / "config.yaml"),
        help="配置文件路径（默认: server/config.yaml）",
    )
    args = parser.parse_args()

    # 1. 加载配置
    print(f"加载配置: {args.config}")
    config = load_config(args.config)

    # 1.5 检测是否启用预览（有 preview 节 = 启用，无 = 关闭）
    preview_cfg = config.get("preview")

    # 2. 初始化日志
    log_cfg = config.get("log", {})
    logger = setup_logger(
        name="smoke_detector",
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/server.log"),
    )
    logger.info("startup smoking_detection_server")

    # 3. 加载模型
    model_path = _PROJECT_ROOT / config["model"]["path"]
    person_cfg = config["model"].get("person_model", {})
    detector = SmokeDetector(
        model_path=model_path,
        conf=config["model"].get("conf", 0.35),
        device=config["model"].get("device", 0),
        person_model_path=person_cfg.get("path"),
        person_conf=person_cfg.get("conf", 0.4),
        imgsz=config["model"].get("imgsz"),
    )

    # 4. 创建 Webhook 推送器（全局共享一个）
    webhook_cfg = config.get("alert", {}).get("webhook", {})
    webhook = None
    if webhook_cfg.get("url"):
        webhook = WebhookAlerter(
            url=webhook_cfg["url"],
            timeout=webhook_cfg.get("timeout", 10),
            retries=webhook_cfg.get("retries", 2),
        )
        logger.debug("Webhook 已配置: {}", webhook_cfg["url"])
    else:
        logger.warning("未配置 Webhook URL，告警将不会推送")

    # 5. 启动摄像头 Workers
    alert_cfg = config.get("alert", {})
    target_classes = config["model"].get("target_classes", ["smoking"])
    workers: list = []
    workers_dict: dict = {}  # preview 模式用

    webhook_url = webhook_cfg.get("url", "none") if webhook else "none"
    for cam in config["cameras"]:
        if not cam.get("enabled", True):
            logger.debug("camera={}:{} disabled", cam["id"], cam.get("name", cam["id"]))
            continue

        logger.info("startup camera={}:{} src={} webhook={} cooldown={}s min_hits={}",
                    cam["id"], cam["name"], cam.get("type", "rtsp"),
                    webhook_url,
                    alert_cfg.get("cooldown_seconds", 30),
                    alert_cfg.get("min_detection_count", 3))

        alert_mgr = AlertManager(
            camera_id=cam["id"],
            camera_name=cam["name"],
            target_classes=target_classes,
            save_frame_overlay=alert_cfg.get("save_frame_overlay", False),
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 30),
            min_detection_count=alert_cfg.get("min_detection_count", 3),
            debug_render=alert_cfg.get("debug_render", False),
            webhook=webhook,
        )

        streamer = _create_streamer(cam)

        worker = CameraWorker(
            camera_id=cam["id"],
            camera_name=cam["name"],
            streamer=streamer,
            detector=detector,
            alert_manager=alert_mgr,
            jpeg_quality=preview_cfg.get("jpeg_quality") if preview_cfg else None,
        )
        if preview_cfg:
            workers_dict[cam["id"]] = worker
            # 不调用 start() — FastAPI lifespan 中注入 _loop 后启动
        else:
            worker.start()

        workers.append(worker)

    if not workers:
        logger.error("没有启用的摄像头，退出")
        return

    # 5.5 启动预览 HTTP 服务器（daemon 线程，不阻塞主线程）
    if preview_cfg:
        _start_preview_server(workers_dict, preview_cfg)

    logger.info("running cameras={}", len(workers))

    # 6. 等待退出信号 + Worker 健康监控
    #    不使用 signal 模块（Windows 下与 GPU 线程交互时有不可靠问题），
    #    改用简单的 KeyboardInterrupt 轮询。
    HEALTH_CHECK_INTERVAL = 30  # 每 30 秒检查一次 Worker 存活状态
    t_last_health = time.time()

    try:
        while True:
            time.sleep(0.5)

            now = time.time()
            if now - t_last_health >= HEALTH_CHECK_INTERVAL:
                for w in workers:
                    if not w.is_running:
                        logger.error(
                            "[{}] Worker 线程已意外退出！请检查日志", w.camera_name
                        )
                t_last_health = now

    except KeyboardInterrupt:
        logger.info("shutdown signal=KeyboardInterrupt")

    # 7. 优雅退出
    for worker in workers:
        worker.stop()
    logger.info("shutdown cameras={}", len(workers))


if __name__ == "__main__":
    main()
