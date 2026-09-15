# 反光条数据增强研究 (reflection_aug)

目标：提升分类模型对**带反光条的反光衣**的区分能力。
数据集：`dataset/manual_cls`（vest=765 / no_vest=324，人体裁剪图 224px）。

## 1. 问题分析

对两类样本做 HSV 统计（`analysis/brightness_stats.py` → `brightness_stats.csv`）：

| class   | n   | V 均值 | V p95 | 高光占比 (V>230) | 条带像素占比 (S<60,V>180) |
|---------|-----|--------|-------|------------------|---------------------------|
| vest    | 765 | 131.4  | 212.9 | 4.6%             | 14.3%                     |
| no_vest | 324 | 112.5  | 192.0 | 2.3%             | 12.9%                     |

结论：
- vest 的判别核心是**反光条**（retro-reflective stripes），其高光/条带像素占比约为 no_vest 的 2 倍；
- 但整体差距不大 → 判别信息集中在**条带的局部表观**，而条带表观随光照剧烈变化：
  - 白天：银灰色哑光条带（低饱和 + 中高亮）
  - 夜间被车灯/手电直射：强烈回射，辉光甚至过曝
  - 夜间无直射光：条带几乎不可见
  - 白色衣物/白色背景与条带在低饱和高亮轴上重叠 → 模型易学错捷径
- 因此增强应覆盖 **条带表观的全光谱**，并抑制"高亮 = vest"的捷径学习。

## 2. 候选增强（`prototypes/reflection_augs.py`，纯 OpenCV/numpy）

| 编号 | 名称 | 模拟场景 | 关键实现 |
|------|------|----------|----------|
| A | night_retroreflection | 夜间反光条被直射光回射 | 整体 gamma 压暗 + HSV 条带 mask 拉亮 + 高斯辉光；mask 面积过大（白衣场景）时自适应衰减 |
| B | lowlight_degrade | 夜间无直射光，条带几乎不可见 | 强压暗 + 高增益噪声 + 蓝色偏移 |
| C | overexposure | 白天强光/AE 拉爆 | gamma 提亮 + 高光软钳制 + bloom |
| D | motion_blur | 视频帧间拖影 | 随机角度方向核卷积 |
| E | flashlight_partial | 局部照明（手电/头灯） | 随机椭圆光斑，亮区去饱和，暗区压黑 |
| F | stripe_washout | 条带老化/污损 | 条带亮度压回衣物中位水平（防捷径） |

效果可视化：`viz/grid_vest.jpg`、`viz/grid_no_vest.jpg`（行=增强，列=随机样本）。

已知局限：
- A 的条带 mask 是颜色启发式（低饱和+高亮），白色衣物会被部分误选（已按面积自适应衰减，且 no_vest 同样被增强，不会引入标签偏置）；
- F 若源图无明显条带（mask 为空）则返回原图。

## 3. 离线增强数据集

`prototypes/augment_offline.py`：

- 读取 `dataset/manual_cls`（扁平 vest/no_vest）；
- 按**同源组**切分 train/val（文件名 `original(N)(K)_pM` 中 `original(N)(K)` 为组，
  同组近重复帧不跨 split，防泄漏；搜索类别比例最接近全局的划分方案）；
- 仅增强 train：每图生成 `--variants` 个变体，每个变体随机选 1 种增强（等概率）；
- val 保持原图不增强（保证评估无偏）。

当前产出 `dataset/manual_cls_aug_split`（默认参数 variants=2, val-ratio=0.25, seed=42）：

- train: vest=574+1148aug, no_vest=243+486aug（合计 2451）
- val: vest=191, no_vest=81（272，干净）
- 6 种增强各 ~270 张，分布均匀

## 4. 训练接入

```powershell
# 完整训练（参数与 scripts/train_vest_cls.py 一致）
uv run python experiments/reflection_aug/prototypes/train_vest_cls_aug.py

# 快速验证
uv run python experiments/reflection_aug/prototypes/augment_offline.py --dry-run
```

已做 sanity 训练（3 epoch, yolo26n-cls, batch256）：链路畅通，val top1 = 0.871，
单 epoch ~1.2s（RTX 5060），300 epoch 约 6 分钟。

## 5. 建议与后续

1. **优先组合**：A/B/C/E 直接对应夜间监控的真实失效模式，预期收益最大；
   F 用于打破"高亮=vest"捷径；D 与 ultralytics 内置增强部分重叠，可保持但权重低。
2. **对照实验**：用相同 seed 分别训 `vest_cls`（无增强）与 `manual_cls_aug_split`（增强），
   比较干净 val 的 top1 及**夜间子集**上的召回（建议从 NoSuit 日期目录抽夜间帧建小测试集）。
3. **变体数消融**：variants ∈ {1, 2, 4}，观察收益饱和点。
4. **若要在线增强**：ultralytics cls 管线不支持自定义增强，可将 `reflection_augs.py`
   挂到自定义 Dataset 的 `__getitem__`，但离线方案简单可控，当前规模（~2.5k 图）无需在线。
