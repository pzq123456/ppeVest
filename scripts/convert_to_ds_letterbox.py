# -*- coding: utf-8 -*-
"""把"拉伸归一化"的分类数据集转换为 DeepStream 风格 (等比缩放 + 黑边对称填充)。

背景:
  DeepStream nvinfer 默认对输入做 letterbox (黑边对称填充), 而本数据集
  (NoSuit_cls_clean_aug_split) 的图片全部是"裁剪 + 直接拉伸到方图"的
  经典 YOLO 方案, 导致训练/部署预处理不一致。

原理:
  对已是正方形的拉伸图直接 letterbox 是无操作 (不会产生黑边), 必须先按
  源裁剪的原始长宽比"逆向还原" (un-stretch), 再补黑边:
      un-stretch(S x S, aspect) -> (w', h')   # 长边 = S, 保持 aspect
      pad black -> S x S
  几何上与 DeepStream 对源裁剪做 symmetric-padding 的输出一致,
  且完整保留图中已烘焙的增强效果 (无需重新生成增强变体)。

两组来源, 长宽比恢复方式不同:
  pre 组  ({idx}_aug{i}.jpg, 320x320):
      源裁剪来自 NoSuit_cls_clean (原始比例), 通过内容检索恢复 aspect:
      48x48 灰度缩略图 + 水平翻转, 归一化相关度匹配, 低于阈值跳过并记录。
  refl 组 (original(N)(K)_pM[__增强名_vi].jpg, 224x224):
      源裁剪由 prepare_manual_cls.py 从 dataset/manual 的 labelme 标注生成,
      aspect 从 {frame}.json 的第 pi 个 person 框 + 5% padding 精确恢复。

输出:
  {out}/{train,val}/{vest,no_vest}/ 与源数据集同名同划分, 另附
  _conversion_report.txt (统计 + 跳过清单)。

用法:
  python scripts/convert_to_ds_letterbox.py [--thresh 600] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "dataset" / "NoSuit_cls_clean_aug_split"
REF_ROOT = PROJECT_ROOT / "dataset" / "NoSuit_cls_clean"
MANUAL_ROOT = PROJECT_ROOT / "dataset" / "manual"
OUT_ROOT = PROJECT_ROOT / "dataset" / "NoSuit_cls_clean_aug_split_ds"

SPLITS = ("train", "val")
CLASSES = ("vest", "no_vest")

THUMB_SIZE = 48
CORR_THRESH = 600.0  # 48x48 归一化相关度, 强匹配经验范围 700~2200

PRE_RE = re.compile(r"^\d{6}_aug\d+$")
VAR_RE = re.compile(r"__(?P<aug>[A-Z]\w*)_(?P<vi>\d+)$")
BASE_RE = re.compile(r"^(?P<frame>.+)_p(?P<pi>\d+)$")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def list_images(directory: Path):
    if not directory.exists():
        return []
    with os.scandir(directory) as it:
        return [Path(e.path) for e in it
                if e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXTS]


def thumb_vec(img, size=THUMB_SIZE):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    v = t.astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def unstretch_letterbox(img, aspect, size):
    """按 aspect (w/h) 还原长宽比 (长边=size), 再黑边对称填充到 size x size。"""
    h, w = img.shape[:2]
    if aspect >= 1.0:
        nw, nh = size, max(1, int(round(size / aspect)))
    else:
        nw, nh = max(1, int(round(size * aspect))), size
    interp = cv2.INTER_AREA if (nw < w or nh < h) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    bottom, right = size - nh - top, size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


class RefPool:
    """某个类别的参考库: 原始比例裁剪 → (aspect, 缩略图向量/翻转向量)。"""

    def __init__(self, class_dir: Path):
        aspects, vecs = [], []
        for p in list_images(class_dir):
            img = imread_unicode(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < 8 or w < 8:
                continue
            v = thumb_vec(img)
            vf = v.reshape(THUMB_SIZE, THUMB_SIZE)[:, ::-1].ravel()
            aspects.extend([w / h, w / h])
            vecs.extend([v, vf])
        self.aspects = np.array(aspects, dtype=np.float32)
        self.vecs = np.stack(vecs) if vecs else np.zeros((0, THUMB_SIZE * THUMB_SIZE), np.float32)

    def match(self, query):
        if len(self.aspects) == 0:
            return None, 0.0
        scores = self.vecs @ query
        best = int(np.argmax(scores))
        return self.aspects[best], float(scores[best])


def convert_pre(files, pool_cache, report, dry_run):
    """pre 组: 内容检索恢复 aspect → un-stretch → 黑边。"""
    written = skipped = 0
    for f in files:
        img = imread_unicode(f)
        if img is None:
            report.skip(f, "unreadable")
            skipped += 1
            continue
        cls = f.parent.name
        if cls not in pool_cache:
            pool_cache[cls] = RefPool(REF_ROOT / cls)
        aspect, corr = pool_cache[cls].match(thumb_vec(img))
        if aspect is None or corr < CORR_THRESH:
            report.skip(f, f"no_ref_match(corr={corr:.0f})")
            skipped += 1
            continue
        if not dry_run:
            out = OUT_ROOT / f.relative_to(SRC_ROOT)
            out.parent.mkdir(parents=True, exist_ok=True)
            if not imwrite_unicode(out, unstretch_letterbox(img, float(aspect), max(img.shape[:2]))):
                report.skip(f, "write_failed")
                skipped += 1
                continue
        written += 1
    return written, skipped


class ManualAnnotations:
    """dataset/manual 的 labelme person 框缓存: {frame_stem: [box, ...]}。"""

    def __init__(self, manual_root: Path):
        self.manual_root = manual_root
        self.cache = {}

    def person_boxes(self, frame_stem):
        if frame_stem not in self.cache:
            jp = self.manual_root / f"{frame_stem}.json"
            if not jp.exists():
                self.cache[frame_stem] = None
            else:
                d = json.loads(jp.read_text(encoding="utf-8"))
                boxes = []
                for s in d.get("shapes", []):
                    if s.get("label") != "person":
                        continue
                    pts = np.array(s["points"], dtype=float)
                    boxes.append((pts[:, 0].min(), pts[:, 1].min(),
                                  pts[:, 0].max(), pts[:, 1].max()))
                self.cache[frame_stem] = boxes
        return self.cache[frame_stem]


def padded_aspect(box, w, h, padding_ratio=0.05):
    """与 prepare_manual_cls.crop_box 一致的 5% 外扩后的长宽比。"""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
    x1, y1, x2, y2 = max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)
    bw, bh = x2 - x1, y2 - y1
    if bh < 1 or bw < 1:
        return None
    return bw / bh


def convert_refl(files, anno, report, dry_run):
    """refl 组: labelme person 框恢复 aspect → un-stretch → 黑边。"""
    written = skipped = 0
    frame_sizes = {}

    def frame_size(frame_stem):
        if frame_stem not in frame_sizes:
            img = imread_unicode(MANUAL_ROOT / f"{frame_stem}.jpg")
            frame_sizes[frame_stem] = None if img is None else (img.shape[1], img.shape[0])
        return frame_sizes[frame_stem]

    for f in files:
        stem = f.stem
        m = VAR_RE.search(stem)
        if m:
            stem = stem[:m.start()]
        bm = BASE_RE.match(stem)
        if not bm:
            report.skip(f, "bad_stem")
            skipped += 1
            continue
        boxes = anno.person_boxes(bm.group("frame"))
        if boxes is None:
            report.skip(f, "json_missing")
            skipped += 1
            continue
        pi = int(bm.group("pi"))
        if pi >= len(boxes):
            report.skip(f, f"person_index_oob(pi={pi},n={len(boxes)})")
            skipped += 1
            continue
        fs = frame_size(bm.group("frame"))
        if fs is None:
            report.skip(f, "frame_unreadable")
            skipped += 1
            continue
        aspect = padded_aspect(boxes[pi], *fs)
        if aspect is None or not (0.1 <= aspect <= 10.0):
            report.skip(f, f"bad_aspect({aspect})")
            skipped += 1
            continue
        img = imread_unicode(f)
        if img is None:
            report.skip(f, "unreadable")
            skipped += 1
            continue
        if not dry_run:
            out = OUT_ROOT / f.relative_to(SRC_ROOT)
            out.parent.mkdir(parents=True, exist_ok=True)
            if not imwrite_unicode(out, unstretch_letterbox(img, aspect, max(img.shape[:2]))):
                report.skip(f, "write_failed")
                skipped += 1
                continue
        written += 1
    return written, skipped


class Report:
    def __init__(self):
        self.lines = []
        self.counts = {}

    def skip(self, f, reason):
        self.lines.append(f"{f.relative_to(SRC_ROOT).as_posix()}  # {reason}")

    def add(self, key, value):
        self.counts[key] = value

    def save(self, path: Path, thresh, dry_run):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(f"thresh={thresh} dry_run={dry_run}\n")
            for k, v in self.counts.items():
                fp.write(f"{k}: {v}\n")
            fp.write(f"\nskipped ({len(self.lines)}):\n")
            fp.write("\n".join(self.lines) + ("\n" if self.lines else ""))


def main():
    global SRC_ROOT, CORR_THRESH
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC_ROOT))
    parser.add_argument("--out", default=str(OUT_ROOT))
    parser.add_argument("--thresh", type=float, default=CORR_THRESH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    SRC_ROOT = Path(args.src)
    CORR_THRESH = args.thresh
    out_root = Path(args.out)
    report = Report()

    pre_all, refl_all = [], []
    for split in SPLITS:
        for cls in CLASSES:
            files = list_images(SRC_ROOT / split / cls)
            pre = [f for f in files if PRE_RE.match(f.stem)]
            refl = [f for f in files if not PRE_RE.match(f.stem)]
            pre_all += [(split, cls, f) for f in pre]
            refl_all += [(split, cls, f) for f in refl]
            print(f"{split}/{cls}: total={len(files)} pre={len(pre)} refl={len(refl)}")

    pool_cache = {}
    wp, sp = convert_pre([f for _, _, f in pre_all], pool_cache, report, args.dry_run)
    anno = ManualAnnotations(MANUAL_ROOT)
    wr, sr = convert_refl([f for _, _, f in refl_all], anno, report, args.dry_run)

    print(f"\npre 组:  转换 {wp}, 跳过 {sp}")
    print(f"refl 组: 转换 {wr}, 跳过 {sr}")
    print(f"合计:    转换 {wp + wr}, 跳过 {sp + sr} / {wp + wr + sp + sr}")

    report.add("pre_written", wp); report.add("pre_skipped", sp)
    report.add("refl_written", wr); report.add("refl_skipped", sr)
    report_path = out_root / "_conversion_report.txt"
    if not args.dry_run:
        report.save(report_path, CORR_THRESH, args.dry_run)
        print(f"报告写入: {report_path}")
        print(f"数据集写入: {out_root}")


if __name__ == "__main__":
    main()
