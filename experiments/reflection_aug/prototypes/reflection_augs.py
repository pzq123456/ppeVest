# -*- coding: utf-8 -*-
"""针对反光条 (retro-reflective stripes) 的数据增强原型.

全部基于 OpenCV/numpy 实现, 便于离线批量生成, 后续可无缝替换/移植.

候选增强 (对应反光条在不同场景下的真实表观变化):
  A. night_retroreflection  夜间反光: 整体压暗 + 反光条区域高亮辉光 (车灯照射)
  B. lowlight_degrade       低光退化: 条带几乎不可见 (夜间无直射光)
  C. overexposure           过曝/眩光: 白天强光或摄像头 AE 拉爆, 条带细节丢失
  D. motion_blur            运动模糊: 监控视频帧间拖影, 条带纹理被抹掉
  E. flashlight_partial     局部闪光: 只有人体局部被照亮, 其余陷入黑暗
  F. stripe_washout         条带褪色: 反光条老化/污损, 灰白条变淡

每个函数签名统一: (bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray
强度参数均带随机性, rng 由调用方传入以保证可复现.
"""
from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------- 通用工具


def _stripe_mask(bgr: np.ndarray, v_thresh: float = 170.0, s_thresh: float = 70.0,
                 grow: int = 1) -> np.ndarray:
    """低饱和 + 高亮 => 银白反光条候选区域, float32 [0,1]."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    mask = ((s < s_thresh) & (v > v_thresh)).astype(np.float32)
    if grow > 0:
        k = np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, k)
    return mask


def _to_hsv_v(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2], hsv


def _merge_hsv(h_u8, s_f, v_f) -> np.ndarray:
    s_f = np.clip(s_f, 0, 255)
    v_f = np.clip(v_f, 0, 255)
    hsv = np.stack([h_u8.astype(np.float32), s_f, v_f], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------- A 夜间反光


def night_retroreflection(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """整体压暗, 反光条区域强烈反光并带辉光.

    模拟: 夜间监控, 反光条被车灯/手电直射时的镜面回射 (retro-reflection).
    白色衣物/背景也会落入 mask, 因此按条带面积占比自适应衰减, 避免整图拉爆.
    """
    h_u8, s, v, _ = _to_hsv_v(bgr)
    # 1) 在原始亮度上先找条带 (压暗后再找就找不到了)
    strip = _stripe_mask(bgr, v_thresh=rng.uniform(150, 185), s_thresh=rng.uniform(55, 80))

    # 2) 全局压暗: gamma 0.5~0.7（温和版，原 0.25~0.45 会把背景压成纯黑，
    #    现场夜间再暗也有环境光/ISP 增益，不可能全黑），再叠一点蓝色调 (夜视偏冷)
    gamma = rng.uniform(0.5, 0.7)
    v_dark = 255.0 * np.power(v / 255.0, 1.0 / max(gamma, 1e-6))
    v_dark *= rng.uniform(0.85, 1.0)
    s_night = s * rng.uniform(0.6, 0.85)  # 夜间饱和度下降
    out = _merge_hsv(h_u8, s_night, v_dark)

    # 3) 条带面积占比过高 (白衣/白墙场景) 时衰减反光强度, 防止整图拉爆
    area_ratio = float(strip.mean())
    scale = min(1.0, 0.10 / max(area_ratio, 1e-6))

    # 4) 条带区域拉到高亮
    boost_lo = rng.uniform(200, 225)
    v_boost = np.where(strip > 0, boost_lo + (255 - boost_lo) * strip, v_dark)
    v_boost = v_dark + (v_boost - v_dark) * scale
    out = _merge_hsv(h_u8, s_night, v_boost)

    # 5) 辉光: 对高亮图做大核高斯模糊, 与原图按条带扩展 mask 混合
    glow_src = np.where(strip[..., None] > 0, out, 0).astype(np.float32)
    glow = cv2.GaussianBlur(glow_src, (0, 0), sigmaX=rng.uniform(4, 9))
    glow_w = cv2.GaussianBlur(strip, (0, 0), sigmaX=rng.uniform(2, 5))[..., None]
    out = np.clip(out.astype(np.float32) + glow * glow_w * rng.uniform(0.6, 1.0) * scale,
                  0, 255)

    # 6) 轻微噪声 (夜间 ISP 增益)
    noise = rng.normal(0, rng.uniform(3, 8), out.shape).astype(np.float32)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- B 低光退化


def lowlight_degrade(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """整体偏暗 + 噪声, 反光条隐约可辨但人体轮廓必须保留 (夜间无直射光源)。

    温和版：gamma 0.55~0.75（原 0.18~0.32，指数压暗 3~5 倍，整图只剩噪声，
    现实中不存在），噪声同步减半。
    """
    gamma = rng.uniform(0.55, 0.75)
    h_u8, s, v, _ = _to_hsv_v(bgr)
    v_dark = 255.0 * np.power(v / 255.0, 1.0 / gamma) * rng.uniform(0.85, 1.0)
    s_dark = s * rng.uniform(0.5, 0.8)
    out = _merge_hsv(h_u8, s_dark, v_dark).astype(np.float32)

    # 高增益传感器噪声 + 轻微色偏
    noise = rng.normal(0, rng.uniform(4, 8), out.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(out)
    b = cv2.convertScaleAbs(b, alpha=1.0, beta=rng.uniform(2, 8))
    return cv2.merge([b, g, r])


# ---------------------------------------------------------------- C 过曝


def overexposure(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """局部/整体过曝: 高光区被拉爆成纯白, 条带纹理丢失.

    模拟: 白天阳光直射或摄像头自动曝光失误.
    """
    h_u8, s, v, _ = _to_hsv_v(bgr)
    strength = rng.uniform(1.25, 1.7)
    v_bright = np.power(v / 255.0, rng.uniform(0.55, 0.75)) * 255.0 * strength

    # 高光软钳制: v 越接近 255 增益越缓, 形成大片纯白
    clip_lo = rng.uniform(200, 235)
    v_bright = np.where(v_bright > clip_lo,
                        clip_lo + (255 - clip_lo) * (v_bright - clip_lo) / (255 - clip_lo + 1e-6),
                        v_bright)
    s_out = s * rng.uniform(0.5, 0.8)  # 过曝时饱和度普遍下降
    out = _merge_hsv(h_u8, s_out, v_bright)

    # 轻微 bloom
    bright_src = np.where((v_bright > 235)[..., None], out, 0).astype(np.float32)
    bloom = cv2.GaussianBlur(bright_src, (0, 0), sigmaX=rng.uniform(3, 7))
    out = np.clip(out.astype(np.float32) * 0.85 + bloom * 0.35, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- D 运动模糊


def motion_blur(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """方向性运动模糊, 模拟监控视频帧间拖影 (反光条细纹理被抹掉)."""
    ksize = rng.integers(7, 17)
    if ksize % 2 == 0:
        ksize += 1
    angle = rng.uniform(0, 180)
    k = np.zeros((ksize, ksize), np.float32)
    c, sn = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    mid = ksize // 2
    for i in range(ksize):
        x = int(round(mid + (i - mid) * c))
        y = int(round(mid + (i - mid) * sn))
        if 0 <= x < ksize and 0 <= y < ksize:
            k[y, x] = 1.0
    k /= k.sum()
    return cv2.filter2D(bgr, -1, k)


# ---------------------------------------------------------------- E 局部闪光


def flashlight_partial(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机位置椭圆照明区, 其余压暗.

    模拟: 手电/头灯只照亮人体局部, 反光条可能落在亮区也可能落在暗区.
    """
    h, w = bgr.shape[:2]
    # 光斑中心/半径随机
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    ax, ay = rng.uniform(0.25, 0.55) * w, rng.uniform(0.3, 0.65) * h

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    spot = np.exp(-d2 * 1.5).astype(np.float32)  # 中心1 边缘0
    spot = cv2.GaussianBlur(spot, (0, 0), sigmaX=rng.uniform(5, 15))

    # 温和版：gamma 0.45~0.65（原 0.15~0.3，暗区直接归零），且暗区保留
    # 原图 35% 亮度下限（手电没照到的地方是暗，不是黑）。
    gamma = rng.uniform(0.45, 0.65)
    h_u8, s, v, _ = _to_hsv_v(bgr)
    v_dark = np.maximum(255.0 * np.power(v / 255.0, 1.0 / gamma), v * 0.35)
    gain = rng.uniform(1.2, 1.5)
    v_out = v_dark * (1 - spot) + np.clip(v * gain, 0, 255) * spot
    s_out = s * (1 - spot * 0.4)  # 亮区去饱和
    out = _merge_hsv(h_u8, s_out, v_out)
    return out


# ---------------------------------------------------------------- F 条带褪色


def stripe_washout(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """反光条淡化/污损: 亮度压回衣物本体水平, 边缘柔化.

    模拟: 反光条老化、灰尘覆盖, 迫使模型不能只依赖高亮条带这一条捷径.
    """
    strip = _stripe_mask(bgr, v_thresh=165, s_thresh=75)
    if strip.sum() == 0:
        return bgr
    h_u8, s, v, _ = _to_hsv_v(bgr)
    # 目标亮度: 图像中位亮度附近 (≈衣物本体)
    target = np.median(v) * rng.uniform(0.9, 1.15)
    keep = rng.uniform(0.25, 0.55)  # 保留多少条带亮度
    v_new = np.where(strip > 0, v * keep + target * (1 - keep), v)
    s_new = np.where(strip > 0, s * keep, s)  # 饱和度同比例压低, 保持灰白
    out = _merge_hsv(h_u8, s_new, v_new)
    # 边缘羽化, 避免生硬
    strip_soft = cv2.GaussianBlur(strip, (0, 0), 2)[..., None]
    out = (out.astype(np.float32) * strip_soft + bgr.astype(np.float32) * (1 - strip_soft))
    return np.clip(out, 0, 255).astype(np.uint8)


AUGS = {
    "A_night_retroreflection": night_retroreflection,
    "B_lowlight_degrade": lowlight_degrade,
    "C_overexposure": overexposure,
    "D_motion_blur": motion_blur,
    "E_flashlight_partial": flashlight_partial,
    "F_stripe_washout": stripe_washout,
}
