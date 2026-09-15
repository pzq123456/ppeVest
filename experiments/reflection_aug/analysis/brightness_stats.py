# -*- coding: utf-8 -*-
"""分析 manual_cls 两类样本的亮度/反光条特征分布.

统计指标:
  - 整体亮度 (V 均值)
  - 高光像素占比 (V > 230): 反光条/过曝区域
  - 低饱和高亮像素占比 (S < 60 且 V > 180): 银白色反光条特征
输出 CSV 到 analysis/.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "dataset" / "manual_cls"
OUT_CSV = Path(__file__).resolve().parent.parent / "analysis" / "brightness_stats.csv"


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def stats_of(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total = v.size
    return {
        "v_mean": float(v.mean()),
        "v_p95": float(np.percentile(v, 95)),
        "specular_ratio": float((v > 230).sum() / total),
        "stripe_like_ratio": float(((s < 60) & (v > 180)).sum() / total),
    }


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    for cls in ("vest", "no_vest"):
        files = sorted((DATA_ROOT / cls).glob("*.jpg"))
        for f in files:
            img = imread_unicode(f)
            if img is None:
                continue
            st = stats_of(img)
            rows.append({"class": cls, "file": f.name, **st})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 汇总
    print(f"{'class':<10}{'n':>6}{'v_mean':>10}{'v_p95':>10}{'spec>230':>10}{'stripe':>10}")
    for cls in ("vest", "no_vest"):
        sub = [r for r in rows if r["class"] == cls]
        print(
            f"{cls:<10}{len(sub):>6}"
            f"{np.mean([r['v_mean'] for r in sub]):>10.1f}"
            f"{np.mean([r['v_p95'] for r in sub]):>10.1f}"
            f"{np.mean([r['specular_ratio'] for r in sub]):>10.4f}"
            f"{np.mean([r['stripe_like_ratio'] for r in sub]):>10.4f}"
        )
    print(f"\n明细写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
