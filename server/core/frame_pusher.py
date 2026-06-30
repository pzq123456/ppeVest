"""
帧推送客户端
- 全局共享一个实例，所有摄像头复用
- 将 FramePacket 列表序列化为 VFb1 二进制协议
- HTTP POST 到对方提供的接收地址
"""

import struct
import urllib.request
import urllib.error
import urllib.parse

from loguru import logger

from server.core.frame_batcher import FramePacket

# VestFrameBatch v1 Magic
MAGIC = b"VFb1"


class FramePushClient:
    """
    帧推送 HTTP 客户端。

    用法:
        client = FramePushClient(url="http://partner:8080/api/vest-frames")
        ok = client.send(camera_id, camera_name, seq, packets, quality)
    """

    def __init__(self, url: str, timeout: float = 10, retries: int = 2):
        """
        Args:
            url: 对方提供的接收地址
            timeout: 单次请求超时（秒）
            retries: 失败后重试次数（不含首次）
        """
        self.url = url
        self.timeout = timeout
        self.retries = retries

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def send(
        self,
        camera_id: str,
        camera_name: str,
        seq: int,
        packets: list[FramePacket],
        jpeg_quality: int = 70,
    ) -> bool:
        """
        打包并 POST 一批帧到对方服务器。

        Returns:
            True 表示发送成功
        """
        if not packets:
            return True

        # 序列化为 VFb1 二进制
        body = self._encode(packets, len(packets), jpeg_quality)

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Camera-Id": camera_id,
            "X-Camera-Name": urllib.parse.quote(camera_name, safe=""),
            "X-Batch-Seq": str(seq),
        }

        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, data=body, headers=headers, method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if 200 <= resp.status < 300:
                        logger.debug(
                            "帧推送成功 cam={} seq={} frames={} size={:.1f}KB",
                            camera_id, seq, len(packets), len(body) / 1024,
                        )
                        return True
                    else:
                        logger.warning(
                            "帧推送返回非 2xx (attempt={}/{}, status={})",
                            attempt + 1, self.retries + 1, resp.status,
                        )

            except urllib.error.HTTPError as e:
                logger.warning(
                    "帧推送 HTTP 错误 (attempt={}/{}, status={}): {}",
                    attempt + 1, self.retries + 1, e.code, e.reason,
                )
            except urllib.error.URLError as e:
                logger.warning(
                    "帧推送连接错误 (attempt={}/{}): {}",
                    attempt + 1, self.retries + 1, e.reason,
                )
            except Exception as e:
                logger.error(
                    "帧推送未知错误 (attempt={}/{}): {}",
                    attempt + 1, self.retries + 1, e,
                )

        logger.error("帧推送失败（已达最大重试次数）cam={} seq={}", camera_id, seq)
        return False

    # ------------------------------------------------------------------
    # VFb1 编码
    # ------------------------------------------------------------------
    @staticmethod
    def _encode(packets: list[FramePacket], count: int, quality: int) -> bytes:
        """
        编码为 VestFrameBatch v1 二进制格式。

        批次头 (8 bytes):
            magic:   4 bytes  "VFb1"
            count:   2 bytes  uint16 BE
            flags:   1 byte   reserved
            quality: 1 byte   JPEG quality

        每帧 (12 + jpeg_len bytes):
            timestamp: 8 bytes  int64 BE (ms)
            data_len:  4 bytes  uint32 BE
            jpeg_data: data_len bytes
        """
        parts = [struct.pack(">4sHBB", MAGIC, count, 0, quality)]
        for pkt in packets:
            parts.append(struct.pack(">qI", pkt.timestamp_ms, len(pkt.jpeg_bytes)))
            parts.append(pkt.jpeg_bytes)
        return b"".join(parts)
