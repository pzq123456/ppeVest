"""
从 NoSuit 告警帧构建反光衣分类数据集。

流程:
  1. 对每个告警帧跑 yolo26n.pt 人体检测 (COCO class 0)
  2. 裁剪人体区域 (含少量 padding)
  3. 用强分类模型 yolo26mcls_ppeVest 对每个 crop 分类，top1 即标签
  4. 按日期文件夹做时间切分 (train/val/test)，输出到 dataset/NoSuit_cls/

注意:
  - 告警帧本身带有平台 OSD 绘制的框/标签，属场景固有噪声，保留。
  - 同一告警事件常跨连续多帧，按日期切分可减少时间泄漏。

用法:
    python scripts/prepare_nosuit_cls.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOSUIT_ROOT = PROJECT_ROOT / "dataset" / "NoSuit"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "NoSuit_cls"

PERSON_MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
CLS_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "yolo26mcls_ppeVest_20260813_1424"
    / "weights"
    / "best.pt"
)

PERSON_CONF = 0.35
PADDING_RATIO = 0.05
MIN_CROP_SIZE = 30
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


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


def assign_split(dates_sorted, val_frac, test_frac):
    """按日期顺序时间切分: train → val → test。"""
    n = len(dates_sorted)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test
    split = {}
    for i, d in enumerate(dates_sorted):
        if i < n_train:
            split[d] = "train"
        elif i < n_train + n_val:
            split[d] = "val"
        else:
            split[d] = "test"
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(NOSUIT_ROOT), help="NoSuit 根目录")
    parser.add_argument("--output", default=str(OUTPUT_ROOT), help="输出数据集根目录")
    parser.add_argument("--person-conf", type=float, default=PERSON_CONF)
    parser.add_argument("--min-size", type=int, default=MIN_CROP_SIZE)
    parser.add_argument("--val-frac", type=float, default=VAL_FRACTION)
    parser.add_argument("--test-frac", type=float, default=TEST_FRACTION)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if not PERSON_MODEL_PATH.exists():
        sys.exit(f"人体检测模型不存在: {PERSON_MODEL_PATH}")
    if not CLS_MODEL_PATH.exists():
        sys.exit(f"分类模型不存在: {CLS_MODEL_PATH}")

    nosuit_root = Path(args.input)
    output_root = Path(args.output)

    # 1. 收集按日期组织的告警帧
    date_dirs = sorted(
        [d for d in nosuit_root.iterdir() if d.is_dir()]
    )
    if not date_dirs:
        sys.exit(f"未在 {nosuit_root} 找到任何日期目录")
    split_of = assign_split([d.name for d in date_dirs], args.val_frac, args.test_frac)

    frame_files = []
    for d in date_dirs:
        for f in sorted(d.glob("*.jpg")):
            frame_files.append(f)
    print(f"告警帧总数: {len(frame_files)}")
    for s in ("train", "val", "test"):
        cnt = sum(1 for f in frame_files if split_of[f.parent.name] == s)
        print(f"  {s}: {cnt} 帧")

    # 2. 加载模型
    print(f"加载人体检测模型: {PERSON_MODEL_PATH}")
    person_model = YOLO(str(PERSON_MODEL_PATH))
    print(f"加载分类模型: {CLS_MODEL_PATH}")
    cls_model = YOLO(str(CLS_MODEL_PATH))
    print(f"分类类别: {cls_model.names}")

    # 3. 逐帧: 检测人体 → 裁剪 → 分类
    stats = {"vest": 0, "no_vest": 0}
    per_split = {"train": {"vest": 0, "no_vest": 0},
                 "val": {"vest": 0, "no_vest": 0},
                 "test": {"vest": 0, "no_vest": 0}}
    no_person = 0

    for i, frame_path in enumerate(frame_files):
        split_name = split_of[frame_path.parent.name]
        img = imread_unicode(frame_path)
        if img is None:
            print(f"  跳过无法读取: {frame_path}")
            continue
        h, w = img.shape[:2]

        results = person_model.predict(
            img, conf=args.person_conf, classes=[0], verbose=False,
            device=args.device,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            no_person += 1
            continue

        boxes_data = boxes.data.cpu().numpy()
        for bi, box in enumerate(boxes_data):
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                continue

            # 小幅外扩，避免贴边裁切
            pw = int((x2 - x1) * PADDING_RATIO)
            ph = int((y2 - y1) * PADDING_RATIO)
            cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
            cx2, cy2 = min(w, x2 + pw), min(h, y2 + ph)

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # 强模型分类 → top1 作为标签
            cls_results = cls_model.predict(crop, verbose=False, device=args.device)
            if cls_results[0].probs is None:
                continue
            top1 = int(cls_results[0].probs.top1)
            label = cls_model.names[top1]

            out_dir = output_root / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{frame_path.parent.name}_{frame_path.stem}_p{bi}"
            if imwrite_unicode(out_dir / f"{stem}.jpg", crop):
                stats[label] += 1
                per_split[split_name][label] += 1

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{len(frame_files)} 帧 | "
                  f"vest={stats['vest']} no_vest={stats['no_vest']}")

    print("\n裁剪统计:")
    print(f"  无人体帧: {no_person}")
    print(f"  合计 vest={stats['vest']}  no_vest={stats['no_vest']}")
    for s in ("train", "val", "test"):
        v, n = per_split[s]["vest"], per_split[s]["no_vest"]
        print(f"  {s}: vest={v} no_vest={n}")

    # 4. 写 data.yaml
    yaml_content = f"""# Vest binary classification built from NoSuit alarm frames
# 推理时取 vest 概率即可: P(vest) = output[1]
path: {output_root.as_posix()}
train: train
val: val
test: test
nc: 2
names: ['no_vest', 'vest']
"""
    (output_root / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\n配置写入: {output_root / 'data.yaml'}")
    print("完成。")


if __name__ == "__main__":
    main()