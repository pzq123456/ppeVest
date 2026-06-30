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
    4. 为每个 enabled 的摄像头创建 CameraWorker
    5. 等待退出信号，优雅关闭
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
from server.core.camera_worker import CameraWorker
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

    # 2. 初始化日志
    log_cfg = config.get("log", {})
    logger = setup_logger(
        name="smoke_detector",
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/server.log"),
    )
    logger.info("=" * 60)
    logger.info("抽烟检测服务端 启动中...")

    # 3. 加载模型
    model_path = _PROJECT_ROOT / config["model"]["path"]
    detector = SmokeDetector(
        model_path=model_path,
        conf=config["model"].get("conf", 0.35),
        device=config["model"].get("device", 0),
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
        logger.info("Webhook 已配置: {}", webhook_cfg["url"])
    else:
        logger.warning("未配置 Webhook URL，告警将不会推送")

    # 5. 启动摄像头 Workers
    alert_cfg = config.get("alert", {})
    target_classes = config["model"].get("target_classes", ["smoking"])
    workers: list[CameraWorker] = []

    for cam in config["cameras"]:
        if not cam.get("enabled", True):
            logger.info("[{}] 已禁用，跳过", cam.get("name", cam["id"]))
            continue

        alert_mgr = AlertManager(
            camera_id=cam["id"],
            camera_name=cam["name"],
            target_classes=target_classes,
            save_frame_overlay=alert_cfg.get("save_frame_overlay", False),
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 30),
            min_detection_count=alert_cfg.get("min_detection_count", 3),
            webhook=webhook,
        )

        streamer = _create_streamer(cam)

        worker = CameraWorker(
            camera_id=cam["id"],
            camera_name=cam["name"],
            streamer=streamer,
            detector=detector,
            alert_manager=alert_mgr,
        )
        worker.start()
        workers.append(worker)

    if not workers:
        logger.error("没有启用的摄像头，退出")
        return

    logger.info("已启动 {} 路摄像头，运行中... (Ctrl+C 停止)", len(workers))

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
        logger.info("收到 Ctrl+C，准备退出...")

    # 7. 优雅退出
    logger.info("正在停止所有 Worker...")
    for worker in workers:
        worker.stop()
    logger.info("服务端已退出")


if __name__ == "__main__":
    main()
