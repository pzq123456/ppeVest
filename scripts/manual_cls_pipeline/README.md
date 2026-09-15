# 手工标注 → 背心分类数据集（标准流水线 v2）

后续每一轮手工标注都按此执行。本目录是唯一入口，老 `scripts/prepare_manual_cls.py` 仅保留作历史对照，不再直接使用。

## 1. 数据流

```text
dataset/manual (+ manual_9_15, manual_9_XX ...)   dataset/manual_hardneg/{vest,no_vest}
  (labelme .jpg + .json)                          (现场 GT 难例裁剪图, 已是 224 letterbox)
        │  prepare_manual_cls_v2.py（扁平，黑边 letterbox；--extra-crops 回灌难例）
        ▼
dataset/manual_cls/{vest,no_vest}      ← 多轮合并后的全量扁平池
        │  augment_offline.py（同源组防泄漏切分 + 仅 train 增强 + 类别均衡）
        ▼
dataset/manual_cls_aug_split/{train,val}/{vest,no_vest} + data.yaml
        │  train_vest_cls.py（imgsz=224，yolo26n-cls，F_beta(no_vest) 选型）
        ▼
runs/.../weights/best_f2.pt            ← 候选权重
        │  eval_vest_cls.py（no_vest 召回门禁）
        ▼
PASS → 部署 DeepStream；FAIL → 回灌难例/调阈值，不要靠加 epoch 硬修
```

## 2. 标注形态（脚本按单文件自动分支）

| 形态 | 特征 | 处理 |
|------|------|------|
| A（老，如 `manual`） | 有 `person` 全身框 + `vest/no_vest` 躯干框 | 躯干中心落入 person → 类别；多框取多数票、平票跳过 |
| B（新，如 `manual_9_15`） | 只有 `vest/no_vest` 人体框 | 该框即直接裁剪源，无匹配 |

## 3. 全局规范（DeepStream 对齐，不可改）

* `PADDING_RATIO=0.05` 外扩，边界夹取，最小边 16px。
* `letterbox_square`：等比缩放到长边=224，短边对称填纯黑 → 224×224，**禁止拉伸**。
* 输出命名：`{frame_stem}_p{idx}.jpg`（如 `2168_..._f1_p0.jpg`，保留 `_fN` 可溯源）。
* 手工补充图（如 `image copy N.jpg`，224×224，不匹配 `*_pM.jpg`）无法从标注再生成，
  脚本重建时自动透传保留（先收集旧输出中的非正则文件，重建后拷回）。
* 防泄漏：`augment_offline.py` 按**同源组**整体切分 train/val，组键三层折叠：
  剥 `_pM` → 剥 `_fN`（事件级，`f1~f5` 同组）→ 折叠 `original(N)(k)` 为 `original(N)`（近重复副本）。
  **2026-09-15 修正**：旧版漏掉第三层，导致 11 个 `original(N)` 家族跨 split（22 张 val 图泄漏）。

## 4. 标准命令（新一轮如 `manual_9_XX` 到货后）

```powershell
# 0) 把新标注放到 dataset/manual_9_XX/（jpg + 同名 labelme json）
#    现场 GT 确认的漏报/误报裁剪图放到 dataset/manual_hardneg/{vest,no_vest}/

# 1) 质检（只读不写，先看数）
python scripts/manual_cls_pipeline/prepare_manual_cls_v2.py --qc-only

# 2) 全量重建扁平池（默认合并 manual + manual_9_15，并回灌 manual_hardneg）
python scripts/manual_cls_pipeline/prepare_manual_cls_v2.py
python scripts/manual_cls_pipeline/prepare_manual_cls_v2.py --inputs dataset/manual dataset/manual_9_15 dataset/manual_9_XX --output dataset/manual_cls

# 3) 增强切分 + 类别均衡（--balance 1.0 默认开启，少数类过采样到与多数类相等）
python experiments/reflection_aug/prototypes/augment_offline.py

# 4) 训练（默认 epochs 500 / patience 150 / beta 2，按 F_beta(no_vest) 选型）
python scripts/manual_cls_pipeline/train_vest_cls.py

# 5) 门禁：no_vest 召回 @0.5 必须 >= 0.95，否则不许上线
python scripts/manual_cls_pipeline/eval_vest_cls.py --model runs/.../weights/best_f2.pt
```

## 4b. 选型与门禁（2026-09-15 SOP 修订，血泪教训）

* **不要用 top1 选模型**：train 里 vest 占 68.7% 时，top1 会掩盖 no_vest 漏报。
  实测同一轮训练：top1 选出的 `best.pt` acc=0.952 但 no_vest 召回仅 0.904（门禁 FAIL）；
  F_beta(no_vest) 选出的 `best_f2.pt` acc=0.943 但 no_vest 召回 0.952（PASS）。
* **也不要只用 no_vest 召回选**：会选出「全判 no_vest」的退化模型（实测 epoch 5 召回 0.971 但 vest 全废）。
  用 `--beta 2` 的 F_beta 同时约束误报。
* **门禁**：`eval_vest_cls.py` 以 `no_vest` 召回@0.5 为准（漏报 > 误报）。未过门禁不得替换现役模型。
* 阈值可调：门禁脚本会给「在 vest 召回 >= 0.85 前提下最大化 no_vest 召回」的推荐阈值。
* 现场失败样本必须回灌 `dataset/manual_hardneg/` 重建，**不要靠加 epoch 硬修**。

## 5. 增强纪律（2026-09-15 修订：增强可做可不做，不过分）

* 本流水线真正重要的只有黑边 letterbox 对齐（DeepStream 一致）；增强是可选的轻扰动。
* 六种增强均为温和版（B 低光 / E 局部光 / A 夜视的压暗指数已下调，不再出现纯黑图）。
* 护栏：变体灰度均值 < `--min-mean 25` 自动重抽，3 次不达标跳过该变体（只留原图）。
* 每图默认只产 1 个变体，合成占比不超过 ~50%。

## 6. 验收清单

* [ ] QC 有效框数 ≈ 各源 `vest+no_vest` 之和（老源按匹配后计），`too_small/conflict_skip` 接近 0。
* [ ] `manual_cls/{vest,no_vest}` 数量符合预期（如 manual 816/327 + 9_15 97/90 → 913/417）。
* [ ] 抽查 20 张：224×224、有黑边、无拉伸、标签正确。
* [ ] `manual_cls_aug_split` 同源组 0 跨 split（含 `original(N)(k)` 折叠），train 类别已均衡。
* [ ] `data.yaml` 为 `nc:2 names:['no_vest','vest']`。
* [ ] **门禁**：`eval_vest_cls.py` no_vest 召回@0.5 ≥ 0.95（PASS）。

## 7. 为什么补充到 `manual_cls`（同类型归属结论，2026-09-15）

* `manual_cls`：DS 对齐 letterbox 224，当前主训练源 —— 新手工标注与之**同分布同预处理**，归这里。
* `vest_cls/vest_cls2`：老拉伸 resize，非 DS 对齐 —— 不进。
* `NoSuit_cls_clean*`：伪标签混合集 —— 不进。
