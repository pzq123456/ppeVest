"""
Generate vest classification dataset from dataset/manual (labelme format).

标注结构:
  - person: 全身框（中位 120x280）
  - vest / no_vest: 躯干框（中位 82x114），中心落入某 person 框 → 该 person 的类别

Output: dataset/manual_cls/{vest,no_vest}/
  - 人体裁剪 letterbox 到 224x224（等比缩放 + 黑边对称填充，与 DeepStream 预处理一致）
  - 扁平结构，train/val 划分由 experiments/reflection_aug/prototypes/augment_offline.py 完成
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_ROOT = PROJECT_ROOT / "dataset" / "manual"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "manual_cls"

CROP_SIZE = 224
PADDING_RATIO = 0.05


def parse_shapes(d: dict) -> tuple[list, list]:
    """解析 labelme shapes → (person_boxes, cls_boxes)。cls_boxes: [(label, box)]"""
    persons, cls_boxes = [], []
    for s in d["shapes"]:
        pts = np.array(s["points"], dtype=float)
        box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
        if s["label"] == "person":
            persons.append(box)
        elif s["label"] in ("vest", "no_vest"):
            cls_boxes.append((s["label"], box))
    return persons, cls_boxes


def center_in(inner: tuple, outer: tuple) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def letterbox_square(crop, size):
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def crop_box(img, box: tuple, w: int, h: int):
    x1, y1, x2, y2 = (int(v) for v in box)
    pad_w, pad_h = int((x2 - x1) * PADDING_RATIO), int((y2 - y1) * PADDING_RATIO)
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return letterbox_square(crop, CROP_SIZE)


def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
        print(f"Removed old: {OUTPUT_ROOT}\n")

    # ---------- 解析全部标注 ----------
    records = []  # (img_path, stem, label, box)
    n_no_label = n_multi = 0
    for img_path in sorted(MANUAL_ROOT.glob("*.jpg")):
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        d = json.loads(json_path.read_text(encoding="utf-8"))
        persons, cls_boxes = parse_shapes(d)

        for pi, pb in enumerate(persons):
            # 找中心落入该 person 的躯干框
            labels = [lbl for lbl, cb in cls_boxes if center_in(cb, pb)]
            if not labels:
                n_no_label += 1
                continue
            if len(labels) > 1:  # 冲突时取多数，平票跳过
                vals, counts = np.unique(labels, return_counts=True)
                if counts.max() > 1:
                    label = vals[counts.argmax()]
                else:
                    n_multi += 1
                    continue
            else:
                label = labels[0]
            records.append((img_path, img_path.stem, label, pb, pi))

    print(f"persons 总数={len(records) + n_no_label + n_multi}, 有效={len(records)}, "
          f"无标签={n_no_label}, 标签冲突跳过={n_multi}")

    # ---------- 裁剪保存（扁平 vest/no_vest，划分交给 augment_offline.py） ----------
    total = defaultdict(int)
    img_cache: dict[Path, np.ndarray] = {}
    for img_path, stem, label, box, pi in records:
        if img_path not in img_cache:
            img_cache.clear()
            img_cache[img_path] = cv2.imread(str(img_path))
        img = img_cache[img_path]
        if img is None:
            continue
        crop = crop_box(img, box, img.shape[1], img.shape[0])
        if crop is None:
            continue
        out_dir = OUTPUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_p{pi}.jpg"), crop)
        total[label] += 1

    print("\nDataset Summary")
    for label in ("vest", "no_vest"):
        print(f"  {label}: {total[label]}")
    n = sum(total.values())
    if n:
        print(f"  vest 比例: {total['vest'] / n:.1%}")


if __name__ == "__main__":
    main()
