"""
后处理：去除 NoSuit 告警帧裁剪样本中的黄色/红色 OSD 遮罩，修正被误导的标签。

背景:
  NoSuit 告警帧来自平台 OSD 渲染，画面中绘制了黄色人物框 + 红色高亮遮罩。
  裁剪人体时这些 OSD 元素被带入 crop，导致强分类模型把无背心误判为有背心。
  本脚本用 inpainting 抹平黄/红像素，得到"干净"的 crop 后重新分类打标签。

用法:
    python scripts/remove_yellow_osd.py
    python scripts/remove_yellow_osd.py --input dataset/NoSuit_cls --output dataset/NoSuit_cls_clean
    python scripts/remove_yellow_osd.py --inpaint-radius 1 --dilation-iters 1
    python scripts/remove_yellow_osd.py --min-color-frac 0.005 --dry-run
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "NoSuit_cls"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "NoSuit_cls_clean"

CLS_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "yolo26mcls_ppeVest_20260813_1424"
    / "weights"
    / "best.pt"
)

# 黄色 HSV 阈值 (OpenCV H:0-180)
YELLOW_H_MIN, YELLOW_H_MAX = 15, 45
YELLOW_S_MIN = 100
YELLOW_V_MIN = 120

# 红色 HSV 阈值 (OpenCV H 红在 0 附近)
RED_H_MAX = 10
RED_H_MIN = 165
RED_S_MIN = 80
RED_V_MIN = 100

# 低于该色占比则不做 inpainting（避免把大块真实彩色物体误抹平）
MIN_COLOR_FRAC = 0.005


def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    try:
        ok, encoded = cv2.imencode(".jpg", img)
        if ok:
            encoded.tofile(str(path))
            return True
    except Exception:
        pass
    return False


def build_color_mask(img, dilation_iters=1, kernel_size=3):
    """检测黄色 + 红色 OSD 像素，返回扩张后的抹平 mask。

    Args:
        dilation_iters: 扩张迭代次数（越小越保守，保留更多原始像素）
        kernel_size: 结构元尺寸（3=细，5=粗）
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)

    yellow = (H >= YELLOW_H_MIN) & (H <= YELLOW_H_MAX) & (S >= YELLOW_S_MIN) & (V >= YELLOW_V_MIN)
    red = ((H <= RED_H_MAX) | (H >= RED_H_MIN)) & (S >= RED_S_MIN) & (V >= RED_V_MIN)

    mask = (yellow | red).astype(np.uint8) * 255
    # 形态学处理: 去噪 + 扩张覆盖抗锯齿边缘
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=dilation_iters)
    return mask


def remove_osd_colors(img, inpaint_radius=1, dilation_iters=1, kernel_size=3):
    """检测黄/红 OSD 像素并用 inpainting 抹平。

    Returns:
        (fixed_img, color_frac, removed_px)
    """
    h, w = img.shape[:2]
    mask = build_color_mask(img, dilation_iters=dilation_iters, kernel_size=kernel_size)
    color_frac = (mask > 0).mean()
    if color_frac < MIN_COLOR_FRAC:
        return img, float(color_frac), 0

    fixed = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return fixed, float(color_frac), int((mask > 0).sum())


def process_dataset(input_root, output_root, args):
    """遍历裁剪数据集，去除黄色/红色 OSD 遮罩并重新分类打标签。

    split 全部合并输出到 output_root/{label}/，不再区分 train/val/test。
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    cls_model = YOLO(str(CLS_MODEL_PATH))
    print(f"分类模型: {CLS_MODEL_PATH}")
    print(f"类别: {cls_model.names}")

    total_crops = 0
    total_processed = 0  # 有遮罩被抹平
    stats = {}
    label_flips = {}

    # 收集所有 crop（合并所有 split）
    crop_paths = []
    for split in ("train", "val", "test"):
        for label_dir in sorted(input_root.glob(f"{split}/*")):
            if label_dir.is_dir():
                crop_paths += sorted(label_dir.glob("*.jpg"))

    print(f"待处理 crop 总数: {len(crop_paths)}")

    for idx, src_path in enumerate(crop_paths):
        parts = src_path.relative_to(input_root).parts  # (split, label, filename)
        split_name, old_label, fname = parts[0], parts[1], parts[2]

        img = imread_unicode(src_path)
        if img is None:
            continue

        fixed, color_frac, removed_px = remove_osd_colors(
            img, inpaint_radius=args.inpaint_radius,
            dilation_iters=args.dilation_iters, kernel_size=args.kernel_size)

        # 用强模型重新分类（处理后的干净 crop）
        results = cls_model.predict(fixed, verbose=False, device=args.device)
        if results[0].probs is None:
            continue
        new_label = cls_model.names[int(results[0].probs.top1)]

        stats.setdefault(new_label, 0)
        stats[new_label] += 1

        if removed_px > 0:
            total_processed += 1
            if old_label != new_label:
                label_flips.setdefault("flips", 0)
                label_flips["flips"] += 1

        # 保存（保持原始尺寸，不改变裁剪范围）
        out_dir = output_root / new_label
        out_dir.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(out_dir / fname, fixed):
            print(f"  写入失败: {src_path}")

        total_crops += 1
        if (idx + 1) % 500 == 0:
            print(f"  已处理 {idx + 1}/{len(crop_paths)}")

    print("\n处理完成:")
    print(f"  总 crop: {total_crops}, 含彩色遮罩被抹平: {total_processed}")
    print(f"  合并后统计: {stats}")
    print(f"  标签翻转: {label_flips}")

    # 写 data.yaml
    yaml_content = f"""# Vest binary classification built from NoSuit alarm frames (OSD colors removed)
# 推理时取 vest 概率即可: P(vest) = output[1]
path: {output_root.as_posix()}
train: .
nc: 2
names: ['no_vest', 'vest']
"""
    (output_root / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"配置写入: {output_root / 'data.yaml'}")


def main():
    global MIN_COLOR_FRAC
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="输入裁剪数据集目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出数据集目录")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--min-color-frac", type=float, default=MIN_COLOR_FRAC)
    parser.add_argument("--inpaint-radius", type=int, default=1)
    parser.add_argument("--dilation-iters", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = parser.parse_args()

    MIN_COLOR_FRAC = args.min_color_frac

    if not CLS_MODEL_PATH.exists():
        sys.exit(f"分类模型不存在: {CLS_MODEL_PATH}")
    if not Path(args.input).exists():
        sys.exit(f"输入目录不存在: {args.input}")

    process_dataset(args.input, args.output, args)
    print("完成。")


if __name__ == "__main__":
    main()