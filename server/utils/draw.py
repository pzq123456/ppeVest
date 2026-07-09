"""
工业风 OSD（On-Screen Display）渲染器
- 纯函数，无业务逻辑依赖，不修改入参
- 三级视觉层级：主实体框 → 属性子框 → 内联标签
- 设计参考 NVIDIA DeepStream NvOSD 管线
"""

from __future__ import annotations

import cv2
import numpy as np

from server.core.detector import ObjectMeta, AttributeMeta


def render_osd(
    frame: np.ndarray,
    objects: list[ObjectMeta],
    *,
    style: str = "preview",
) -> np.ndarray:
    """在帧副本上叠加工业风 OSD 标注。

    Args:
        frame: BGR 图像 (H, W, 3)
        objects: 检测元数据列表
        style: 渲染风格
            - "preview": 绿框主体 + 黄框属性子框 + 紧凑内联标签（实时预览用）
            - "alert":   红框主体 + SMOKING 标签 + 黄框属性子框（告警证据帧用）
            - "debug":   预览风格 + 全部置信度 + 属性详情

    Returns:
        标注后的新图像（始终 copy，不修改入参）
    """
    annotated = frame.copy()
    for obj in objects:
        if style == "alert":
            _draw_alert(annotated, obj)
        elif style == "debug":
            _draw_debug(annotated, obj)
        else:
            _draw_preview(annotated, obj)
    return annotated


# ======================================================================
# 逐风格渲染器（内部函数）
# ======================================================================

def _draw_preview(frame: np.ndarray, obj: ObjectMeta):
    """预览模式：绿框主体 2px + 黄框属性子框 1px + 紧凑内联标签。"""
    x1, y1, x2, y2 = obj.bbox
    color = (0, 255, 0)

    # Tier 1: 主实体框 2px 实线
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Tier 3: 内联标签（实体 + 属性拼接为单行）
    label = f"{obj.class_name} {obj.confidence:.2f}"
    for attr in obj.attributes:
        label += f" [{attr.name.upper()} {attr.confidence:.2f}]"
    cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Tier 2: 属性子框 1px（仅带空间坐标的属性）
    for attr in obj.attributes:
        if attr.bbox is not None:
            ax1, ay1, ax2, ay2 = attr.bbox
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 255), 1)
            cv2.putText(frame, f"{attr.name} {attr.confidence:.2f}",
                        (ax1, max(ay1 - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def _draw_alert(frame: np.ndarray, obj: ObjectMeta):
    """告警模式：红框主体 2px + 动态告警大写标签 + 黄框属性子框 1px。

    标签优先取 classification 属性的名称（bbox=None），
    否则取 detection 属性的名称，兜底为 "ALERT"。
    """
    x1, y1, x2, y2 = obj.bbox

    # 从属性中派生告警标签：优先取纯分类属性（如 vest/no_vest），其次取检测属性
    _cls_attrs = [a.name.upper() for a in obj.attributes if a.bbox is None]
    _det_attrs = [a.name.upper() for a in obj.attributes if a.bbox is not None]
    label = _cls_attrs[0] if _cls_attrs else (_det_attrs[0] if _det_attrs else "ALERT")

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Tier 2: 属性子框 1px 黄色实线
    for attr in obj.attributes:
        if attr.bbox is not None:
            ax1, ay1, ax2, ay2 = attr.bbox
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 255), 1)
            cv2.putText(frame, f"{attr.name} {attr.confidence:.2f}",
                        (ax1, max(ay1 - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def _draw_debug(frame: np.ndarray, obj: ObjectMeta):
    """调试模式：预览风格 + 属性详情展开。"""
    x1, y1, x2, y2 = obj.bbox

    # Tier 1: 主实体框 2px 实线
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 标签：实体置信度
    cv2.putText(frame, f"{obj.class_name} {obj.confidence:.2f}",
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 每个属性展开为独立标签行（垂直偏移避免重叠）
    offset = 0
    for attr in obj.attributes:
        offset += 18
        label = f"  {attr.name} {attr.confidence:.2f}"
        if attr.bbox is not None:
            ax1, ay1, ax2, ay2 = attr.bbox
            label += f" @({ax1},{ay1})-({ax2},{ay2})"
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 255), 1)
        cv2.putText(frame, label,
                    (x1 + 5, max(y1 - 8 + offset, 16 + offset)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
