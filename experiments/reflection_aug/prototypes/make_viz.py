# -*- coding: utf-8 -*-
"""对样本图应用各候选增强, 生成网格对比图到 viz/.

每张网格: 行=增强种类 (第0行为原图), 列=不同随机样本, 便于评估强度是否合理.
用法:
    uv run python experiments/reflection_aug/prototypes/make_viz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reflection_augs import AUGS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "dataset" / "manual_cls"
VIZ_ROOT = Path(__file__).resolve().parents[1] / "viz"

N_COLS = 4  # 每种增强展示的随机样本数
SHOW_SIZE = 160


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", img)
    if ok:
        encoded.tofile(str(path))


def cell(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (SHOW_SIZE, SHOW_SIZE), interpolation=cv2.INTER_AREA)


def main() -> None:
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    VIZ_ROOT.mkdir(parents=True, exist_ok=True)

    for cls in ("vest", "no_vest"):
        files = sorted((DATA_ROOT / cls).glob("*.jpg"))
        rng = np.random.default_rng(42)
        picks = rng.choice(len(files), size=N_COLS, replace=False)

        originals = [imread_unicode(files[i]) for i in picks]
        rows = [[cell(im) for im in originals]]
        labels = ["original"]

        for name, fn in AUGS.items():
            row = []
            for ci, im in enumerate(originals):
                if im is None:
                    row.append(np.zeros((SHOW_SIZE, SHOW_SIZE, 3), np.uint8))
                    continue
                row.append(cell(fn(im, np.random.default_rng(1000 + ci * 77))))
            rows.append(row)
            labels.append(name)

        # 拼网格 + 行标签
        pad, label_w = 4, 190
        grid_h = len(rows) * (SHOW_SIZE + pad) + pad
        grid_w = label_w + N_COLS * (SHOW_SIZE + pad) + pad
        canvas = np.full((grid_h, grid_w, 3), 32, np.uint8)
        for r, row in enumerate(rows):
            y = pad + r * (SHOW_SIZE + pad)
            cv2.putText(canvas, labels[r], (6, y + SHOW_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            for c, imc in enumerate(row):
                x = label_w + pad + c * (SHOW_SIZE + pad)
                canvas[y:y + SHOW_SIZE, x:x + SHOW_SIZE] = imc

        out = VIZ_ROOT / f"grid_{cls}.jpg"
        imwrite_unicode(out, canvas)
        print(f"写入 {out}")


if __name__ == "__main__":
    main()
