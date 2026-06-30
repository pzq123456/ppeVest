# 反光衣检测服务端

基于 YOLO 的**级联模型**（检测→分类）反光衣实时检测后台服务。支持多路 RTSP / 本地摄像头接入，检测到目标后通过 Webhook 推送告警，并通过 HTTP 二进制协议实时推送标注帧。

## 目录结构

```
ppeVest/
├── server/
│   ├── main.py                 # 入口
│   ├── config.yaml             # 配置文件（修改此处即可，无需改代码）
│   ├── core/
│   │   ├── streamer.py         # 视频流读取器（RTSP + 本地摄像头）
│   │   ├── detector.py         # YOLO 模型封装（single / cascade 双模式）
│   │   ├── camera_worker.py    # 单路摄像头 Worker 线程
│   │   ├── frame_batcher.py    # 帧批量收集器（时间窗口打包）
│   │   └── frame_pusher.py     # 帧推送客户端（VFb1 二进制编码 + HTTP POST）
│   ├── alert/
│   │   ├── webhook.py          # Webhook HTTP POST 推送
│   │   └── manager.py          # 告警管理（冷却 + 连续帧确认 + 帧标注编码）
│   ├── utils/
│   │   └── logger.py           # loguru 日志配置
│   └── README.md
├── local/
│   └── frame_receiver.py       # 帧推送测试接收端（MJPEG 预览 + VFb1 解析）
└── pyproject.toml
```

## 快速开始

```bash
# 安装依赖（uv）
uv sync

# 终端 1：启动帧推送接收器（模拟消费端，保存帧 + 提供 MJPEG 预览）
python local/frame_receiver.py --port 8080

# 终端 2：启动检测服务
python -m server.main

# 指定配置文件
python -m server.main --config my_config.yaml
```

## 级联模型架构

```
视频帧 → YOLO Detect (找人, class 0) → 裁切人体区域
                                         ↓
                                    YOLO Classify
                                    vest / no_vest
                                         ↓
                              ┌──────────┴──────────┐
                              ↓                      ↓
                        AlertManager              FrameBatcher
                     (Webhook 告警推送)         (帧批量打包推送)
```

**模式切换：** 修改 `config.yaml` 中 `model.type` 字段：
- `"cascade"` — 级联模式：检测 + 分类（当前）
- `"single"` — 单模型模式：一个 YOLO 模型直接检测（原抽烟检测兼容）

## 配置文件参考 (config.yaml)

### model — 模型参数

#### cascade 模式

| 字段 | 类型 | 说明 |
|------|------|------|
| `model.type` | string | `"cascade"` |
| `model.device` | int/string/null | GPU 设备 ID；`0`=第一块GPU；`null`或`"cpu"`=CPU |
| `model.detect.model` | string | 人体检测模型；纯文件名（如 `yolo11n.pt`）自动下载 |
| `model.detect.conf` | float | 检测置信度阈值 |
| `model.detect.target_class_id` | int | 目标类别 ID，COCO person = 0 |
| `model.classify.path` | string | 分类模型权重路径（相对项目根目录） |
| `model.classify.conf` | float | 分类置信度阈值 |
| `model.classify.labels` | dict | 类别标签映射，如 `{0: "no_vest", 1: "vest"}` |
| `model.target_classes` | list | 触发告警的目标类别 |

#### single 模式

| 字段 | 类型 | 说明 |
|------|------|------|
| `model.type` | string | `"single"` |
| `model.path` | string | YOLO 模型权重路径 |
| `model.conf` | float/dict | 置信度阈值；dict 格式支持逐类设定 |

### cameras — 摄像头列表

每路摄像头是一个数组元素：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | string | **必填** | 唯一标识，用于告警 payload 和帧推送 |
| `name` | string | **必填** | 显示名称，用于日志和告警 |
| `type` | string | `"rtsp"` | 摄像头类型：`rtsp`（RTSP网络流）或 `local`（本地USB/内置摄像头） |
| `enabled` | bool | `true` | 是否启用该摄像头 |

**type=rtsp 时额外字段：**

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `rtsp_url` | string | **是** | RTSP 流地址 |

**type=local 时额外字段：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device_id` | int | `0` | OpenCV 摄像头设备 ID，`0`=默认摄像头 |

### alert — 告警参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cooldown_seconds` | float | `30` | 同一摄像头两次告警的最小间隔（秒） |
| `min_detection_count` | int | `3` | 连续检测到目标的帧数阈值，防止单帧误报 |
| `save_frame_overlay` | bool | `false` | 是否在证据帧上叠加摄像头名称/时间水印 |
| `require_all_targets` | bool | 自动 | 是否要求所有 target_classes 同时存在。单类别默认 `false`，多类别默认 `true`。显式指定优先 |

#### alert.webhook — Webhook 推送

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | string | `null` | Webhook 接收地址。为 null 时不推送 |
| `timeout` | float | `10` | 单次请求超时（秒） |
| `retries` | int | `2` | 失败后重试次数（不含首次） |

### frame_push — 帧推送

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用帧推送 |
| `url` | string | **必填** | 对方提供的帧接收地址 |
| `batch_interval_ms` | int | `1000` | 批次间隔（毫秒），值越小延迟越低但请求频率越高 |
| `max_frames_per_batch` | int | `15` | 单批次帧数上限 |
| `jpeg_quality` | int | `70` | JPEG 编码质量（1-100） |
| `timeout` | float | `10` | HTTP 请求超时（秒） |
| `retries` | int | `2` | 失败后重试次数 |

### log — 日志参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `file` | string | `"logs/server.log"` | 日志文件路径（JSON 格式，自动轮转 10MB，保留 30 天，旧文件 gz 压缩） |

## 告警流程

```
读取帧 → 级联推理 → 检测到目标类别？
                         ↓ 是
                   连续帧计数器 +1
                         ↓
                   达到 min_detection_count？
                         ↓ 是
                   冷却期已过？
                         ↓ 是
                   🚨 触发告警
                   ├── 标注帧（bbox + 可选水印）
                   ├── JPEG 编码 → base64
                   └── 构建 payload → Webhook POST
                                        ↓
                              接收端消费
                              ├── 解码 base64 → 写入磁盘
                              └── 结构化记录推送数据
```

**关键设计：**
- **连续帧确认**：只有连续 `min_detection_count` 帧（默认 3 帧）都检测到才触发，杜绝单帧噪点误报
- **冷却期**：同一摄像头 `cooldown_seconds` 秒内只告警一次，避免告警风暴
- **计数器递减**：没有检测到目标时，连续计数器逐步递减（而非直接清零），容忍偶尔丢帧
- **base64 证据帧**：标注后的 JPEG 直接编码进 payload，接收端无需访问检测端文件系统
- **检测端不写磁盘**：帧保存由消费端负责，检测端只推送

## Webhook Payload 格式

```json
{
  "camera_id": "gate",
  "camera_name": "测试",
  "timestamp": "2026-06-30T09:30:00+00:00",
  "detections": [
    {
      "class": "no_vest",
      "confidence": 0.92,
      "bbox": [320, 240, 400, 380]
    }
  ],
  "frame_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

## 帧推送协议

帧通过 HTTP POST 以二进制格式推送到对方提供的地址。

### 端点

```
POST {frame_push.url}
Content-Type: application/octet-stream
```

### Headers

| Header | 示例 | 说明 |
|--------|------|------|
| `X-Camera-Id` | `gate` | 摄像头唯一标识 |
| `X-Camera-Name` | `%E6%B5%8B%E8%AF%95` | 摄像头名称（URL 编码） |
| `X-Batch-Seq` | `42` | 批次序号，单调递增，用于检测丢包 |

### Body（VestFrameBatch v1）

```
Offset  Size  Field
------  ----  -----
0       4     Magic: "VFb1" (0x56 0x46 0x62 0x31)
4       2     frame_count: uint16 big-endian
6       1     flags: reserved (0x00)
7       1     jpeg_quality: 1-100

--- 以下重复 frame_count 次 ---
8+N*12  8     timestamp_ms: int64 big-endian (Unix ms)
16+N*12 4     jpeg_len: uint32 big-endian
20+N*12 N     jpeg_data: JPEG 二进制
```

**解析伪代码：**
```
offset = 0
magic, count, flags, quality = read(offset, 8)
offset += 8
for i in range(count):
    ts, jpeg_len = read(offset, 12)
    offset += 12
    jpeg = read(offset, jpeg_len)
    offset += jpeg_len
```

## 测试工具

### 帧推送接收器

`local/frame_receiver.py` 模拟第三方消费端：

```bash
# 终端1：启动接收器（默认 0.0.0.0:8080）
python local/frame_receiver.py --port 8080

# 终端2：启动检测服务（config.yaml 中 frame_push.url 设为 http://localhost:8080/api/vest-frames）
python -m server.main
```

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` / `-p` | `8080` | 监听端口 |
| `--save-dir` | `frames_received` | 帧保存目录 |

端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/vest-frames` | POST | 接收 VFb1 二进制帧批次 |
| `/stream/{camera_id}` | GET | MJPEG 实时预览（浏览器打开） |
| `/stats` | GET | 接收统计 JSON |
| `/` | GET | 摄像头列表（含预览链接） |

接收端文件输出结构：

```
frames_received/
└── gate/
    ├── 20260630_164500_123.jpg
    └── 20260630_164501_456.jpg
```

### 本地摄像头测试

修改 `config.yaml`，启用本地摄像头配置段即可在不连接 RTSP 流的情况下测试。

## 日志级别说明

| 级别 | 内容 | 频率 |
|------|------|------|
| INFO | 启动/停止、告警触发、Webhook 推送、帧推送、每 60 秒运行摘要 | 低频 |
| DEBUG | 每 100 帧的详细统计（FPS、推理延迟）、检测命中、帧推送详情 | 高频 |
| WARNING | RTSP 断线重连、帧读取失败、帧推送失败 | 按需 |
| ERROR | 检测异常、Webhook 推送失败、Worker 意外退出 | 按需 |

生产环境推荐使用 `INFO` 级别，调试时使用 `DEBUG`。

## 稳定性设计

### 多层异常防护

每个线程都有顶层 try/except 安全网，确保单个异常不会杀死线程：

```
streamer._update_loop()     ← 顶层 try/except，异常后 1s 恢复
camera_worker._run()        ← 顶层 try/except，异常后跳过当前帧继续
alert_manager.handle()      ← 顶层 try/except，异常后返回 False
frame_pusher.send()         ← 内部重试 + 异常捕获
webhook.send()              ← 内部重试 + 异常捕获
```

### RTSP 断线重连

RTSP 流断开后自动重连，使用指数退避策略（初始 2 秒，每次失败翻倍，最大 60 秒），连接成功后重置延迟。

### Worker 健康监控

主线程每 30 秒检查所有 Worker 线程是否存活，发现死亡立即记录 ERROR 日志。

### 优雅退出

Ctrl+C → 主线程捕获 KeyboardInterrupt → 依次 stop 所有 Worker → 释放资源 → 退出。不使用 signal 模块（Windows 下与 GPU 线程交互存在已知可靠性问题）。
