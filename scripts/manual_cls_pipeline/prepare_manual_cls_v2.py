# -*- coding: utf-8 -*-
"""手工标注 -> vest 分类数据集（标准流水线 v2）。

支持两种标注形态（按单文件自动判定，可混合输入）：
  A. 老形态 dataset/manual：person 全身框 + vest/no_vest 躯干框。
     躯干框中心落入某 person 框 -> 该 person 类别；多框冲突取多数票、平票跳过。
  B. 新形态 dataset/manual_9_15：只有 vest/no_vest 人体框，无 person。
     该框即直接裁剪源，无需匹配。

全局规范（DeepStream 对齐，不可改）：
  - person/人体框外扩 PADDING_RATIO=0.05，边界夹取，最小 16px
  - letterbox：等比缩放到长边=224，短边两侧对称填纯黑 -> 224x224，禁止拉伸
  - 输出扁平：{output}/{vest,no_vest}/{frame_stem}_p{idx}.jpg
  - train/val 划分与增强不在此脚本做，交给 augment_offline.py（按同源组防泄漏）

用法（标准）：
  python scripts/manual_cls_pipeline/prepare_manual_cls_v2.py --qc-only
  python scripts/manual_cls_pipeline/prepare_manual_cls_v2.py

  python scripts/manual_cls_pipeline/prepare_manual_cls_v2.py ^
      --inputs dataset/manual dataset/manual_9_15 --output dataset/manual_cls_test
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [PROJECT_ROOT / "dataset" / "manual",
                  PROJECT_ROOT / "dataset" / "manual_9_15"]
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "manual_cls"

CROP_SIZE = 224
PADDING_RATIO = 0.05
MIN_SIDE = 16
VALID_LABELS = ("vest", "no_vest")

DEFAULT_HARDNEG = PROJECT_ROOT / "dataset" / "manual_hardneg"

# 非脚本生成的手工补充图（如 image copy N.jpg）不匹配此正则，重建时透传保留。
REGEN_RE = re.compile(r"^.+_p\d+$")


def load_carry_from(root: Path, size: int) -> list[tuple[str, str, "np.ndarray"]]:
    """从另一版输出（如 224 版 manual_cls）取非脚本生成的补充图并 letterbox 到 size。

    已 letterbox 过的方图再 letterbox 到新尺寸，几何上等价于原图 letterbox 到新尺寸
    （等比缩放的复合仍是等比缩放，黑边同步缩放）。
    """
    out: list[tuple[str, str, "np.ndarray"]] = []
    for cls in VALID_LABELS:
        d = root / cls
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jpg")):
            if REGEN_RE.match(f.stem) or f.name.startswith("hn_"):
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue
            if img.shape[0] != size or img.shape[1] != size:
                img = letterbox_square(img, size)
            out.append((cls, f.name, img))
    return out


def collect_hardneg(root: Path) -> list[tuple[str, Path]]:
    """收集现场 GT 确认的难例裁剪图（已是 224 letterbox）-> [(cls, path)]。

    目录结构：{root}/{vest,no_vest}/*.jpg。这些图来自真实管线
    （检测框 + DS letterbox），与部署分布一致，用于回灌 OOD 难例。
    """
    out: list[tuple[str, Path]] = []
    for cls in VALID_LABELS:
        d = root / cls
        if d.exists():
            for f in sorted(d.glob("*.jpg")):
                out.append((cls, f))
    return out


def collect_carryover(out_root: Path) -> list[tuple[str, str, bytes]]:
    """收集旧输出中无法从标注再生成的补充图 -> [(cls, name, bytes)]。

    注意：必须读入内存（调用后 out_root 会被删除，不可保留 Path 引用）。
    """
    keep: list[tuple[str, str, bytes]] = []
    for cls in VALID_LABELS:
        d = out_root / cls
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jpg")):
            # hn_ 前缀由 --extra-crops 显式摄入，避免重复计入
            if not REGEN_RE.match(f.stem) and not f.name.startswith("hn_"):
                keep.append((cls, f.name, f.read_bytes()))
    return keep


def parse_shapes(d: dict) -> tuple[list, list]:
    """-> (person_boxes, cls_boxes[(label, box)])。box=(x1,y1,x2,y2)。"""
    persons, cls_boxes = [], []
    for s in d.get("shapes", []):
        label = s.get("label")
        if label not in ("person", "vest", "no_vest"):
            continue
        pts = np.array(s["points"], dtype=float)
        box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
        if label == "person":
            persons.append(box)
        else:
            cls_boxes.append((label, box))
    return persons, cls_boxes


def center_in(inner: tuple, outer: tuple) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def resolve_records(img_path: Path, persons: list, cls_boxes: list,
                    stats: dict) -> list:
    """单帧解析 -> [(stem, label, box, pi)]。A/B 形态自动分支。"""
    stem = img_path.stem
    if persons:
        # A 形态：中心点匹配
        out = []
        for pi, pb in enumerate(persons):
            labels = [lbl for lbl, cb in cls_boxes if center_in(cb, pb)]
            if not labels:
                stats["no_label"] += 1
                continue
            if len(labels) > 1:
                vals, counts = np.unique(labels, return_counts=True)
                if counts.max() > 1:
                    label = str(vals[counts.argmax()])
                else:
                    stats["conflict_skip"] += 1
                    continue
            else:
                label = labels[0]
            out.append((stem, label, pb, pi))
        return out
    # B 形态：直接裁剪
    if not cls_boxes:
        stats["empty"] += 1
        return []
    out = []
    for bi, (label, box) in enumerate(cls_boxes):
        w, h = box[2] - box[0], box[3] - box[1]
        stats["direct_box_w"].append(float(w))
        stats["direct_box_h"].append(float(h))
        if w < MIN_SIDE or h < MIN_SIDE:
            stats["too_small"] += 1
            continue
        out.append((stem, label, box, bi))
    return out


def letterbox_square(crop, size: int):
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
    if crop.size == 0 or crop.shape[0] < MIN_SIDE or crop.shape[1] < MIN_SIDE:
        return None
    return letterbox_square(crop, CROP_SIZE)


def collect(inputs: list[Path]) -> tuple[list, dict]:
    """遍历全部输入源 -> records + stats。"""
    records: list = []
    stats: dict = defaultdict(int)
    stats["direct_box_w"] = []
    stats["direct_box_h"] = []
    stats["per_source"] = {}
    for src in inputs:
        n_img, n_rec = 0, 0
        for img_path in sorted(src.glob("*.jpg")):
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                stats["missing_json"] += 1
                continue
            d = json.loads(json_path.read_text(encoding="utf-8"))
            persons, cls_boxes = parse_shapes(d)
            recs = resolve_records(img_path, persons, cls_boxes, stats)
            records.extend(recs)
            n_img += 1
            n_rec += len(recs)
        stats["per_source"][src.name] = (n_img, n_rec)
    return records, stats


def print_qc(records: list, stats: dict) -> None:
    total_v = sum(1 for r in records if r[1] == "vest")
    total_n = len(records) - total_v
    print("=" * 60)
    print("QC 报告（未写盘）")
    print("=" * 60)
    for src, (ni, nr) in stats["per_source"].items():
        print(f"  {src}: {ni} 帧 -> {nr} 有效框")
    print(f"  有效总数={len(records)} (vest={total_v} no_vest={total_n})")
    if records:
        print(f"  vest 占比: {total_v / len(records):.1%}")
    for k in ("no_label", "conflict_skip", "empty", "too_small", "missing_json"):
        if stats.get(k):
            print(f"  {k}: {stats[k]}")
    if stats["direct_box_w"]:
        ws = sorted(stats["direct_box_w"])
        hs = sorted(stats["direct_box_h"])
        print(f"  直接裁剪框(B形态) n={len(ws)} "
              f"w中位={ws[len(ws)//2]:.0f} h中位={hs[len(hs)//2]:.0f} "
              f"(w p10={ws[len(ws)//10]:.0f} h p10={hs[len(hs)//10]:.0f})")


def main() -> None:
    global CROP_SIZE
    ap = argparse.ArgumentParser(description="手工标注->分类数据集标准流水线 v2")
    ap.add_argument("--inputs", nargs="+", default=[str(p) for p in DEFAULT_INPUTS])
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--extra-crops", default=str(DEFAULT_HARDNEG),
                    help="现场 GT 难例裁剪图目录（{vest,no_vest}/*.jpg，已是 letterbox），"
                         "以 hn_ 前缀并入；目录不存在则忽略")
    ap.add_argument("--crop-size", type=int, default=CROP_SIZE,
                    help="letterbox 目标边长（须与 DeepStream 部署一致；基线 0904 为 320）")
    ap.add_argument("--carry-from", default=None,
                    help="从另一版输出（如 dataset/manual_cls）继承手工补充图并 letterbox 到"
                         "--crop-size；切换分辨率时用")
    ap.add_argument("--qc-only", action="store_true", help="只质检不写盘")
    args = ap.parse_args()
    CROP_SIZE = args.crop_size

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            sys.exit(f"输入源不存在: {p}")
    records, stats = collect(inputs)
    print_qc(records, stats)
    if args.qc_only:
        return

    out_root = Path(args.output)
    if args.carry_from:
        carry_src = Path(args.carry_from)
        if not carry_src.exists():
            sys.exit(f"--carry-from 不存在: {carry_src}")
        carry_images = load_carry_from(carry_src, CROP_SIZE)
        carryover = [(cls, name, None) for cls, name, _ in carry_images]
    else:
        carry_images = []
        carryover = collect_carryover(out_root) if not args.qc_only else []
    if carryover:
        print(f"\n手工补充图(透传保留): {len(carryover)} 张")
        for cls, name, _ in carryover[:5]:
            print(f"  {cls}/{name}")
        if len(carryover) > 5:
            print(f"  ... +{len(carryover) - 5}")
    if out_root.exists():
        shutil.rmtree(out_root)
        print(f"Removed old: {out_root}")

    total: dict = defaultdict(int)
    img_cache: dict = {}
    n_crop_fail = 0
    for img_path_stem, label, box, pi in records:
        # 反查原图路径（多源同名极少，优先后输入源覆盖前者不发生：命名空间不同）
        img_path = None
        for src in inputs:
            cand = src / f"{img_path_stem}.jpg"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        if img_path not in img_cache:
            img_cache.clear()
            img_cache[img_path] = cv2.imread(str(img_path))
        img = img_cache[img_path]
        if img is None:
            n_crop_fail += 1
            continue
        crop = crop_box(img, box, img.shape[1], img.shape[0])
        if crop is None:
            n_crop_fail += 1
            continue
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{img_path_stem}_p{pi}.jpg"), crop)
        total[label] += 1

    n_carry = 0
    if carry_images:
        for cls, name, img in carry_images:
            out_dir = out_root / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / name), img)
            total[cls] += 1
            n_carry += 1
    else:
        for cls, name, blob in carryover:
            out_dir = out_root / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / name).write_bytes(blob)
            total[cls] += 1
            n_carry += 1
    if n_carry:
        print(f"\n透传保留手工补充图: {n_carry} 张")

    hn_root = Path(args.extra_crops)
    hardneg = collect_hardneg(hn_root) if hn_root.exists() else []
    if hardneg:
        for cls, f in hardneg:
            out_dir = out_root / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"hn_{f.name}").write_bytes(f.read_bytes())
            total[cls] += 1
        print(f"现场难例回灌({hn_root.name}): {len(hardneg)} 张")

    print("\nDataset Summary")
    for label in VALID_LABELS:
        print(f"  {label}: {total[label]}")
    n = sum(total.values())
    if n:
        print(f"  vest 占比: {total['vest'] / n:.1%}")
    if n_crop_fail:
        print(f"  裁剪失败(读图/过小): {n_crop_fail}")
    print(f"\n输出: {out_root}")


if __name__ == "__main__":
    main()
