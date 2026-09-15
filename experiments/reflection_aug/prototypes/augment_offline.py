# -*- coding: utf-8 -*-
"""离线数据增强: manual_cls → manual_cls_aug_split

流程:
  1. 读取 dataset/manual_cls/{vest,no_vest} (扁平结构)
  2. 按文件名同源组切分 train/val ("original(N)(K)" 为组, _pM 为人体编号,
     同组近重复帧不跨 split, 防泄漏)
  3. 仅对 train 应用本研究的反光条增强 (每图 N 个变体, 每变体随机选 1 种)
  4. 输出 dataset/manual_cls_aug_split/{train,val}/{vest,no_vest} + data.yaml

用法:
    uv run python experiments/reflection_aug/prototypes/augment_offline.py \
        [--variants 2] [--val-ratio 0.25] [--seed 42] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reflection_augs import AUGS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "dataset" / "manual_cls"
OUT_ROOT = PROJECT_ROOT / "dataset" / "manual_cls_aug_split"

GROUP_RE = re.compile(r"^(?P<group>.+)_p\d+$")  # original(1)(2)_p3 → original(1)(2)


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ok, encoded = cv2.imencode(".jpg", img)
    if ok:
        encoded.tofile(str(path))
    return ok


def split_by_group(files: list[Path], val_ratio: float, rng: np.random.Generator):
    """按同源组整体划分, 搜索类别比例最接近全局比例的方案."""
    group_of = {}
    for f in files:
        m = GROUP_RE.match(f.stem)
        group_of[f] = m.group("group") if m else f.stem
    groups = sorted(set(group_of.values()))

    labels = {f: f.parent.name for f in files}
    global_ratio = sum(1 for f in files if labels[f] == "vest") / max(len(files), 1)

    best = None
    for _ in range(500):
        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * val_ratio))
        val_set = set(groups[:n_val])
        val_files = [f for f in files if group_of[f] in val_set]
        if not val_files:
            continue
        ratio = sum(1 for f in val_files if labels[f] == "vest") / len(val_files)
        score = abs(ratio - global_ratio) + abs(len(val_files) / len(files) - val_ratio)
        if best is None or score < best[0]:
            best = (score, val_set)

    val_set = best[1]
    train = [f for f in files if group_of[f] not in val_set]
    val = [f for f in files if group_of[f] in val_set]
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=2, help="每张 train 图生成的增强变体数")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(args.seed)

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
        print(f"Removed old: {OUT_ROOT}\n")

    # 1. 收集
    all_files, per_cls = [], {}
    for cls in ("vest", "no_vest"):
        files = sorted((SRC_ROOT / cls).glob("*.jpg"))
        per_cls[cls] = files
        all_files.extend(files)
    print(f"源图: vest={len(per_cls['vest'])} no_vest={len(per_cls['no_vest'])}")

    # 2. 切分
    train_files, val_files = split_by_group(all_files, args.val_ratio, rng)
    print(f"切分: train={len(train_files)} val={len(val_files)}")
    for s, fs in (("train", train_files), ("val", val_files)):
        v = sum(1 for f in fs if f.parent.name == "vest")
        print(f"  {s}: vest={v} no_vest={len(fs) - v}")

    if args.dry_run:
        return

    # 3. 输出 val (原图, 不增强)
    aug_names = list(AUGS)
    stats = defaultdict(int)
    for f in val_files:
        img = imread_unicode(f)
        if img is None:
            print(f"  无法读取, 跳过: {f.name}")
            continue
        out_dir = OUT_ROOT / "val" / f.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        if imwrite_unicode(out_dir / f.name, img):
            stats[f"val/{f.parent.name}"] += 1

    # 4. 输出 train (原图 + N 个增强变体)
    for f in train_files:
        img = imread_unicode(f)
        if img is None:
            print(f"  无法读取, 跳过: {f.name}")
            continue
        out_dir = OUT_ROOT / "train" / f.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(out_dir / f.name, img)
        stats[f"train/{f.parent.name}"] += 1

        for vi in range(args.variants):
            aug_name = aug_names[rng.integers(len(aug_names))]
            aug_rng = np.random.default_rng(rng.integers(2**63))
            aug_img = AUGS[aug_name](img, aug_rng)
            imwrite_unicode(out_dir / f"{f.stem}__{aug_name}_{vi}.jpg", aug_img)
            stats[f"aug/{aug_name}"] += 1

    # 5. data.yaml
    yaml_content = f"""# Vest binary cls: manual_cls + reflection-strip offline augmentation
# 增强: {", ".join(aug_names)}
path: {OUT_ROOT.as_posix()}
train: train
val: val
nc: 2
names: ['no_vest', 'vest']
"""
    (OUT_ROOT / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print("\n输出统计:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"\n数据集写入: {OUT_ROOT}")


if __name__ == "__main__":
    main()
