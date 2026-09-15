# -*- coding: utf-8 -*-
"""离线数据增强: manual_cls → manual_cls_aug_split

流程:
  1. 读取 dataset/manual_cls/{vest,no_vest} (扁平结构)
  2. 按文件名同源组切分 train/val ("original(N)(K)" 为组, _pM 为人体编号,
     新连拍命名额外剥 _fN 按事件分组, 同组近重复帧不跨 split, 防泄漏)
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
FRAME_RE = re.compile(r"^(?P<event>.+)_f\d+$")  # 2168_..._8922_f1 → 2168_..._8922（连拍事件）
DUP_RE = re.compile(r"^(?P<root>.+\(\d+\))\(\d+\)$")  # original(10)(1) → original(10)（近重复副本）


def group_of_stem(stem: str) -> str:
    """同源组键：剥 _pM → 剥 _fN → 折叠近重复副本，同源组不跨 split。

    - original(1)(2)_p3 → original(1)（老命名；original(N)(k) 是 original(N) 的近重复帧）
    - 2168_..._8922_f1_p0 → 2168_..._8922（新事件级分组）
    - image copy 2（手工补充，无 _pM）→ 自身单例组
    """
    m = GROUP_RE.match(stem)
    base = m.group("group") if m else stem
    fm = FRAME_RE.match(base)
    if fm:
        base = fm.group("event")
    dm = DUP_RE.match(base)
    return dm.group("root") if dm else base


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ok, encoded = cv2.imencode(".jpg", img)
    if ok:
        encoded.tofile(str(path))
    return ok


def make_variant(img: np.ndarray, rng: np.random.Generator, aug_names: list[str],
                 min_mean: float):
    """生成一个增强变体；亮度护栏不过则重抽，最多 3 次；失败返回 None。"""
    for _ in range(3):
        aug_name = aug_names[rng.integers(len(aug_names))]
        aug_rng = np.random.default_rng(rng.integers(2**63))
        aug_img = AUGS[aug_name](img, aug_rng)
        if float(cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY).mean()) >= min_mean:
            return aug_name, aug_img
    return None


def split_by_group(files: list[Path], val_ratio: float, rng: np.random.Generator):
    """按同源组整体划分, 搜索类别比例最接近全局比例的方案."""
    group_of = {}
    for f in files:
        group_of[f] = group_of_stem(f.stem)
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
    global SRC_ROOT, OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=1, help="每张 train 图生成的增强变体数")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-mean", type=float, default=25.0,
                        help="变体灰度均值下限：低于此视为不真实的纯黑图，重抽其他增强，"
                             "3 次都不达标则跳过该变体（只保留原图）")
    parser.add_argument("--balance", type=float, default=1.0,
                        help="train 类别均衡：对少数类过采样增强变体，使其数量占多数类的该比例；"
                             "0 表示关闭。补偿分类器无 class weight 的偏置（默认 1.0）")
    parser.add_argument("--src", default=str(SRC_ROOT), help="扁平源目录（如 320 版数据集）")
    parser.add_argument("--out", default=str(OUT_ROOT), help="输出根目录")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    SRC_ROOT, OUT_ROOT = Path(args.src), Path(args.out)
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
            picked = make_variant(img, rng, aug_names, args.min_mean)
            if picked is None:
                stats["aug/skipped_too_dark"] += 1
                continue
            aug_name, aug_img = picked
            imwrite_unicode(out_dir / f"{f.stem}__{aug_name}_{vi}.jpg", aug_img)
            stats[f"aug/{aug_name}"] += 1

    # 4b. 类别均衡：过采样少数类，补偿 ultralytics 分类器无 class weight 的固有偏置
    if args.balance > 0:
        cls_counts = {c: len(list((OUT_ROOT / "train" / c).glob("*.jpg")))
                      for c in ("vest", "no_vest")}
        mino = min(cls_counts, key=cls_counts.get)
        other = "vest" if mino == "no_vest" else "no_vest"
        target = int(round(cls_counts[other] * args.balance))
        pool = [f for f in train_files if f.parent.name == mino]
        out_dir = OUT_ROOT / "train" / mino
        n_extra = attempts = 0
        while cls_counts[mino] < target and pool and attempts < target * 5:
            f = pool[attempts % len(pool)]
            attempts += 1
            img = imread_unicode(f)
            if img is None:
                continue
            picked = make_variant(img, rng, aug_names, args.min_mean)
            if picked is None:
                stats["aug/skipped_too_dark"] += 1
                continue
            aug_name, aug_img = picked
            imwrite_unicode(out_dir / f"{f.stem}__{aug_name}_bal{n_extra}.jpg", aug_img)
            cls_counts[mino] += 1
            n_extra += 1
        print(f"均衡: 少数类 {mino} 过采样 +{n_extra} -> {cls_counts[mino]} "
              f"(目标 {target}, 多数类 {other}={cls_counts[other]})")

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
