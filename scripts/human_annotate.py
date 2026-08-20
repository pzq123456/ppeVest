"""
人机辅助分类标注工具（GUI）— v2

核心流程:
  1. 强分类模型预计算所有样本的预测建议 + 置信度
  2. 置信度 > AUTO_THRESHOLD (默认 0.9) 的样本自动采纳模型建议，不进入人工流程
  3. 剩余样本逐个显示，默认按 vest 处理（空格=归为 vest）
  4. 方向键 ← → 在样本间自由导航；按 v/n/d 决策并记录
  5. 决策即时生效（移动文件），日志落盘 (.annotate_log.csv) 支持断点续跑

进一步减少工作量:
  - 高置信度自动处理
  - 默认 vest，多数样本只需按空格或右方向键跳过
  - 置信度提示 + 与当前标签不一致时高亮，快速定位需人工的样本

用法:
    python scripts/human_annotate.py
    python scripts/human_annotate.py --auto-threshold 0.9
    python scripts/human_annotate.py --no-model      # 不加载模型，纯人工

快捷键:
    ← → = 上一个/下一个样本（不记录决策）
    Space/Enter = 归为 vest        n = 归为 no_vest
    d = 丢弃（误判/坏样本）          q = 保存退出
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "NoSuit_cls_clean"

CLS_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "yolo26mcls_ppeVest_20260813_1424"
    / "weights"
    / "best.pt"
)

LABELS = ["vest", "no_vest"]
LOG_FILENAME = ".annotate_log.csv"
MAX_DISPLAY_H = 560
MAX_DISPLAY_W = 760


def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class HumanAnnotator:
    def __init__(self, root: tk.Tk, dataset: Path, use_model: bool,
                 auto_threshold: float):
        self.root = root
        self.dataset = dataset
        self.use_model = use_model
        self.auto_threshold = auto_threshold
        self.log_path = dataset / LOG_FILENAME

        self.decisions = []
        self.processed_ids = set()
        self._load_log()

        # 收集样本（排除已决策的）
        self.items = []
        for label in LABELS:
            d = dataset / label
            if d.exists():
                for f in sorted(d.glob("*.jpg")):
                    if str(f) in self.processed_ids:
                        continue
                    self.items.append({
                        "src": f, "cur_label": label,
                        "pred_label": label, "conf": 0.0,
                    })
        self.total_all = len(self.items)

        # 模型建议预计算
        if self.use_model:
            self._precompute_predictions()

        # 自动采纳高置信度样本
        self.auto_done = self._auto_accept()

        # 剩余人工样本
        self.items = [it for it in self.items if not it.get("auto", False)]
        self.idx = 0
        self.total = len(self.items)

        self._build_ui()
        self.root.title(
            f"人机辅助分类标注 - 待确认 {self.total} / 自动处理 {self.auto_done}")
        self._update_display()

    # ------------------------------------------------------------------
    # 预计算与自动采纳
    # ------------------------------------------------------------------
    def _precompute_predictions(self):
        model = YOLO(str(CLS_MODEL_PATH))
        print(f"加载分类模型: {CLS_MODEL_PATH}")
        for i, item in enumerate(self.items):
            img = imread_unicode(item["src"])
            if img is None:
                continue
            try:
                res = model.predict(img, verbose=False)
                probs = res[0].probs
                if probs is not None:
                    item["pred_label"] = model.names[int(probs.top1)]
                    item["conf"] = float(probs.top1conf.item())
            except Exception:
                pass
            if (i + 1) % 500 == 0:
                print(f"  预计算 {i + 1}/{len(self.items)}")

    def _auto_accept(self) -> int:
        if not self.use_model:
            return 0
        count = 0
        for item in self.items:
            if item["conf"] > self.auto_threshold:
                label = item["pred_label"]
                if label != item["cur_label"]:
                    dst = self.dataset / label / item["src"].name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        dst = dst.with_name(f"{item['src'].stem}_a{item['src'].suffix}")
                    shutil.move(str(item["src"]), str(dst))
                    item["src"] = dst
                    item["cur_label"] = label
                item["auto"] = True
                self._record_decision(item, label, moved=True, auto=True)
                count += 1
        if count:
            print(f"自动采纳高置信度样本: {count}")
        return count

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _load_log(self):
        if not self.log_path.exists():
            return
        with open(self.log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.decisions.append(row)
                self.processed_ids.add(row["src"])

    def _append_log(self, row: dict):
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(fill=tk.X)
        self.progress = ttk.Progressbar(top, maximum=max(self.total, 1), value=0)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.count_label = ttk.Label(top, text="0/0", width=10)
        self.count_label.pack(side=tk.RIGHT, padx=(8, 0))

        img_frame = ttk.Frame(self.root)
        img_frame.pack(fill=tk.BOTH, expand=True)
        self.img_label = tk.Label(img_frame, bg="#222")
        self.img_label.pack(fill=tk.BOTH, expand=True)

        info = ttk.Frame(self.root, padding=(12, 4))
        info.pack(fill=tk.X)
        self.file_label = ttk.Label(info, text="", font=("Consolas", 10))
        self.file_label.pack()
        self.pred_label = ttk.Label(info, text="", font=("Arial", 14))
        self.pred_label.pack(pady=(4, 0))

        btn = ttk.Frame(self.root, padding=(12, 10))
        btn.pack(fill=tk.X)
        ttk.Button(btn, text="归为 vest  (Space)",
                   command=lambda: self.decide("vest")).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(btn, text="归为 no_vest  (n)",
                   command=lambda: self.decide("no_vest")).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(btn, text="丢弃  (d)", command=self.discard).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(btn, text="退出  (q)", command=self.save_exit).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self.root.bind("<space>", lambda e: self.decide("vest"))
        self.root.bind("<Return>", lambda e: self.decide("vest"))
        self.root.bind("<n>", lambda e: self.decide("no_vest"))
        self.root.bind("<d>", lambda e: self.discard())
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("<Right>", lambda e: self.next())
        self.root.bind("<q>", lambda e: self.save_exit())

    # ------------------------------------------------------------------
    # 导航与显示
    # ------------------------------------------------------------------
    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._update_display()

    def next(self):
        if self.idx < self.total - 1:
            self.idx += 1
            self._update_display()
        else:
            self._finish()

    def _update_display(self):
        if self.idx >= self.total:
            self._finish()
            return
        item = self.items[self.idx]
        img = imread_unicode(item["src"])
        if img is None:
            self.idx += 1
            self._update_display()
            return

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w, h = pil.size
        scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
        if scale < 1.0:
            pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil)
        self.img_label.config(image=self.photo)

        rel = item["src"].relative_to(self.dataset)
        self.file_label.config(text=f"[{self.idx + 1}/{self.total}]  {rel}")

        if self.use_model:
            conf = item["conf"]
            color = "#2a7d32" if item["pred_label"] == "vest" else "#c62828"
            diff = " ⇐ 模型建议不同" if item["pred_label"] != item["cur_label"] else ""
            self.pred_label.config(
                text=f"当前: {item['cur_label']}    模型: {item['pred_label']}"
                     f" ({conf:.3f}){diff}",
                foreground=color)
        else:
            self.pred_label.config(text=f"当前: {item['cur_label']}", foreground="#333")

        self.progress["value"] = self.idx
        self.count_label.config(text=f"{self.idx + 1}/{self.total}")

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------
    def _move(self, src, label, suffix=""):
        dst = self.dataset / label / src.name
        if dst.exists():
            dst = dst.with_name(f"{src.stem}{suffix}{src.suffix}")
        shutil.move(str(src), str(dst))
        return dst, True

    def decide(self, label: str):
        if self.idx >= self.total:
            return
        item = self.items[self.idx]
        moved = False
        if label != item["cur_label"]:
            new_src, moved = self._move(item["src"], label, suffix="_m")
            item["src"] = new_src
            item["cur_label"] = label
        self._record_decision(item, label, moved=moved, auto=False)
        self.next()

    def discard(self):
        if self.idx >= self.total:
            return
        item = self.items[self.idx]
        discard_dir = self.dataset / "discard"
        discard_dir.mkdir(parents=True, exist_ok=True)
        dst = discard_dir / item["src"].name
        if dst.exists():
            dst = dst.with_name(f"{item['src'].stem}_d{item['src'].suffix}")
        shutil.move(str(item["src"]), str(dst))
        item["src"] = dst
        item["cur_label"] = "discard"
        self._record_decision(item, "discard", moved=True, auto=False)
        self.next()

    def _record_decision(self, item, label, moved: bool, auto: bool):
        row = {
            "src": str(item["src"]),
            "label": label,
            "model_pred": item["pred_label"] if self.use_model else "",
            "conf": f"{item['conf']:.4f}" if self.use_model else "",
            "moved": moved,
            "auto": auto,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        self.decisions.append(row)
        self._append_log(row)

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def _finish(self):
        self.progress["value"] = self.total
        self.count_label.config(text=f"{self.total}/{self.total}")
        self.img_label.config(image="", text="全部完成 ✓", foreground="#2a7d32")
        self.file_label.config(text="")
        self.pred_label.config(
            text=f"自动处理 {self.auto_done}，人工处理 {len(self.decisions)}，"
                 f"日志: {self.log_path}")
        self.root.bind("<q>", lambda e: self.save_exit())

    def save_exit(self):
        self.root.destroy()


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--no-model", action="store_true", help="不加载模型，纯人工")
    parser.add_argument("--auto-threshold", type=float, default=0.85,
                        help="置信度高于该值自动采纳模型建议")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        sys.exit(f"数据集目录不存在: {dataset}")

    root = tk.Tk()
    root.geometry("900x760")
    app = HumanAnnotator(root, dataset, use_model=not args.no_model,
                         auto_threshold=args.auto_threshold)
    root.mainloop()


if __name__ == "__main__":
    main()