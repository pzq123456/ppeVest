# -*- coding: utf-8 -*-
"""标准训练入口：manual_cls_aug_split，以 no_vest 方向性指标选型。

与旧脚本的差别（2026-09-15 SOP 修订）：
  - 旧脚本用 ultralytics 默认 top1 选 best.pt → 在 vest 占多数时掩盖 no_vest 漏报；
  - 本脚本挂 on_fit_epoch_end 回调，用 **F_beta(no_vest)** 选型（默认 beta=2，召回优先，
    对应 SOP「漏报 > 误报」），另存 best_f2.pt，并把每 epoch 曲线写入 metrics.csv。
  - 只按 no_vest 召回单选会选出「全判 no_vest」的退化模型（实测 epoch 5 召回 0.97 但
    vest 全废），故必须用 F_beta 这种同时约束误报的指标。
  - 类别均衡由 augment_offline.py --balance 在数据侧完成（分类器无 class weight）。

用法：
  python scripts/manual_cls_pipeline/train_vest_cls.py [--epochs 500 --patience 150 --beta 2]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset" / "manual_cls_aug_split"


class DirectionalSelector:
    """按 F_beta(no_vest) 保存最佳权重，并记录 nv_recall / vest_recall / f_beta 曲线。"""

    def __init__(self, nv_index: int = 0, beta: float = 2.0):
        self.best = -1.0
        self.best_epoch = -1
        self.beta = beta
        self.nv_index = nv_index
        self.log_path: Path | None = None

    @staticmethod
    def _resolve_index(validator, default: int) -> int:
        names = getattr(validator, "names", None)
        if isinstance(names, dict):
            for k, v in names.items():
                if v == "no_vest":
                    return int(k)
        return default

    def __call__(self, trainer) -> None:
        validator = getattr(trainer, "validator", None)
        cm = getattr(validator, "confusion_matrix", None) if validator else None
        if cm is None:
            return
        if self.log_path is None:
            self.log_path = Path(trainer.save_dir) / "metrics.csv"
            self.log_path.write_text("epoch,no_vest_recall,vest_recall,f_beta\n", encoding="utf-8")

        c = self._resolve_index(validator, self.nv_index)
        v = 1 - c if c in (0, 1) else 0
        m = np.asarray(cm.matrix, dtype=float)
        tp, fn = m[c, c], m[v, c]      # matrix[pred][true]
        fp, tn = m[c, v], m[v, v]
        nv_rec = tp / (tp + fn) if (tp + fn) else 0.0
        v_rec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        b2 = self.beta ** 2
        fbeta = (1 + b2) * prec * nv_rec / (b2 * prec + nv_rec) if (prec + nv_rec) else 0.0

        with open(self.log_path, "a", encoding="utf-8") as fp_out:
            fp_out.write(f"{trainer.epoch},{nv_rec:.6f},{v_rec:.6f},{fbeta:.6f}\n")

        trainer.metrics = trainer.metrics or {}
        trainer.metrics["metrics/fbeta_novest"] = fbeta

        if fbeta > self.best:
            self.best, self.best_epoch = fbeta, int(trainer.epoch)
            trainer.save_model()
            dst = Path(trainer.wdir) / "best_f2.pt"
            dst.write_bytes(Path(trainer.last).read_bytes())
            try:
                from ultralytics.utils.torch_utils import strip_optimizer
                strip_optimizer(str(dst))
            except Exception:
                pass
            print(f"  [f_beta] epoch {trainer.epoch}: nv_recall={nv_rec:.4f} "
                  f"vest_recall={v_rec:.4f} f_beta={fbeta:.4f} -> {dst.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="背心分类标准训练（F_beta 方向性选型）")
    ap.add_argument("--model", default="yolo26n-cls.pt")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--beta", type=float, default=2.0, help="F_beta 的 beta，越大越偏召回")
    ap.add_argument("--device", default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from ultralytics import YOLO

    model_tag = Path(args.model).stem.replace("-cls", "").replace(".pt", "")
    run_name = args.name or f"{model_tag}_vest_cls_recall_{datetime.now():%Y%m%d_%H%M}"
    selector = DirectionalSelector(beta=args.beta)
    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", selector)

    model.train(
        data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=4,
        project=str(PROJECT_ROOT / "runs"), name=run_name,
    )

    save_dir = selector.log_path.parent if selector.log_path else PROJECT_ROOT / "runs" / run_name
    weights = save_dir / "weights" / "best_f2.pt"
    print(f"\n选型完成: best F_beta={selector.best:.4f} @epoch {selector.best_epoch}")
    print(f"权重: {weights}")
    print(f"门禁: python scripts/manual_cls_pipeline/eval_vest_cls.py --model {weights}")


if __name__ == "__main__":
    main()
