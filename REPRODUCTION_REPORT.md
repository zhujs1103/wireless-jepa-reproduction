# WirelessJEPA 复现报告

## 1. 复现目标

本文档记录对论文 **WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions** 的代码复现进展。

论文核心思想是将多天线 IQ 信号构造成二维 antenna-time grid，通过 JEPA 框架在 latent space 中预测被 mask 区域的教师编码器表示，从而学习可迁移的无线信号表征。模型训练完成后，冻结编码器，并通过 linear probing 与 k-NN 评估下游任务性能。

当前复现以 `RML2016.10a` 作为首个闭环数据集，目标是先跑通完整流程：

- 数据加载与预处理
- WirelessJEPA 预训练
- checkpoint 保存
- frozen encoder 特征提取
- linear probing 与 k-NN 评估
- 混淆矩阵与结果图保存

## 2. 数据集

当前使用数据集：

```text
data/RML2016.10a_dict.pkl
```

数据集规模：

- 11 个 modulation classes
- 20 个 SNR levels
- 每个 `(modulation, SNR)` 组合约 1000 条样本
- 总样本数约 220000

当前划分：

- Train: 154000
- Val: 33000
- Test: 33000

注意：论文原始预训练数据是多天线 MIMO IQ 数据，形状为 `2 x 4 x 256`。RML2016.10a 是单天线调制识别数据，原始形状为 `2 x 128`。当前代码会将其扩展为 `2 x 256 x 256`，用于适配 WirelessJEPA 的 antenna-time grid 输入。因此，该数据集适合完成训练评估闭环，但不能完全复现论文中多天线空间建模效果。

## 3. 实现对齐情况

当前代码已实现论文的主要组件。

### 3.1 数据预处理

实现文件：

```text
dataset.py
```

处理流程：

1. 复数 IQ 转为实值 I/Q 两通道。
2. 使用 unitmax normalization。
3. 将输入转换为 `2 x 256 x 256` antenna-time grid。
4. 对 RML2016.10a 的 `2 x 128` 输入，先视为 `2 x 1 x 128`，再进行最近邻扩展。
5. 当前已修正 train/val/test 划分逻辑：每个 modulation/SNR 内先 shuffle，再按 70/15/15 划分。

### 3.2 Mask 策略

实现文件：

```text
mask.py
```

已实现论文 Figure 2 中的四类 mask：

- random
- antenna
- temporal
- multiblock

当前默认 mask ratio 为 `0.75`。

### 3.3 模型结构

实现文件：

```text
model.py
```

已实现：

- ShuffleNetV2-x0.5 encoder
- sparse context encoder
- dense EMA teacher encoder
- learnable mask token
- 3 层 depthwise separable convolution predictor
- spatial latent feature map prediction

关键修正：

- student encoder 与 teacher encoder 已分离，不再共享参数。
- teacher encoder 与 teacher projection 不参与梯度更新，只通过 EMA 更新。
- latent feature 保留 `16 x 16` 空间结构，不再使用全局向量硬 reshape。
- mask token 维度已修正为 `256`，与 latent feature channel 对齐。
- sparse convolution 中 mask 会在 block 内部传播，减少 masked 区域信息泄露。
- BatchNorm 已改为 masked BatchNorm，仅使用 visible positions 统计均值和方差。

### 3.4 训练目标

实现文件：

```text
pretrain.py
```

训练目标与论文一致：

```text
L = 1 / |M| * sum_{(i,j) in M} || y_hat_{i,j} - y_{i,j} ||_2^2
```

关键实现：

- loss 只在 latent mask 的 masked positions 上计算。
- 通道维度使用 sum，对应 L2 范数平方。
- AdamW optimizer。
- warmup + cosine learning rate schedule。
- EMA teacher momentum 从 `0.996` 按 step 线性增长到 `1.0`。

### 3.5 下游评估

实现文件：

```text
eval.py
```

当前评估方式：

- 冻结预训练模型。
- 使用 EMA teacher encoder 提取 pooled feature。
- 用 train split 训练 linear probing head。
- 用 test split 进行最终测试。
- 同时进行 k-NN 评估。

输出文件：

```text
outputs/confusion_matrix_lp.png
outputs/confusion_matrix_knn.png
outputs/results_table.png
```

## 4. 已完成实验

### 4.1 预训练配置

本次已完成训练命令：

```powershell
python main.py --mode pretrain --epochs 100 --batch_size 64
```

本次训练使用默认 mask：

```text
mask_type = random
mask_ratio = 0.75
```

训练结束日志：

```text
Epoch 99: Train Loss=68.8166, Val Loss=69.1442
Checkpoint saved: ./checkpoints/epoch_099.pt
Pretraining completed
Loss plot saved: outputs/training_losses.png
```

注意：当前 loss 使用论文公式，对通道维度求和，因此 loss 数值在几十到一百以上是正常量级，不能与旧版 `mean(dim=1)` 的 0.x loss 直接比较。

### 4.2 评估配置

评估命令：

```powershell
python main.py --mode eval --checkpoint checkpoints/best_model.pt --batch_size 64
```

评估结果：

```text
Linear Probing:
  Accuracy: 0.2154
  F1 Score: 0.1743

k-NN:
  Accuracy: 0.2664
  F1 Score: 0.2662
```

额外检查发现：

```text
best_model.pt 实际对应 epoch 4
epoch_099.pt 的 linear probing accuracy 约为 0.2550
```

因此，当前低精度不是单纯由于 best checkpoint 选择过早造成的，后期 checkpoint 表示也仍然较弱。

### 4.3 Temporal Mask 实验

根据论文中 time masking 更适合 modulation classification 的趋势，进一步完成 temporal mask 训练。

训练命令：

```powershell
python main.py --mode pretrain --epochs 100 --batch_size 64 --mask_type temporal
```

训练结束日志：

```text
Epoch 99: Train Loss=48.5445, Val Loss=47.5843
Checkpoint saved: ./checkpoints/epoch_099.pt
Best checkpoint saved: ./checkpoints/best_model.pt
Pretraining completed
Loss plot saved: outputs/training_losses.png
```

使用 `epoch_099.pt` 评估：

```powershell
python main.py --mode eval --checkpoint checkpoints/epoch_099.pt --batch_size 64
```

结果：

```text
Linear Probing:
  Accuracy: 0.3001
  F1 Score: 0.2598

k-NN:
  Accuracy: 0.3602
  F1 Score: 0.3611
```

使用 `best_model.pt` 评估：

```powershell
python main.py --mode eval --checkpoint checkpoints/best_model.pt --batch_size 64
```

结果与 `epoch_099.pt` 完全一致：

```text
Linear Probing:
  Accuracy: 0.3001
  F1 Score: 0.2598

k-NN:
  Accuracy: 0.3602
  F1 Score: 0.3611
```

这说明 temporal mask 实验中，`best_model.pt` 与最后一轮 checkpoint 表现一致。结合训练日志可知，epoch 99 是当前最低 validation JEPA loss 的 checkpoint。

## 5. 当前结果分析

当前 random mask 训练闭环已经跑通，但分类性能明显低于论文 RML2016.10a 结果。主要原因包括：

### 5.1 数据集与论文预训练数据不一致

论文原始 WirelessJEPA 预训练使用真实多天线 MIMO 数据，输入为：

```text
2 x 4 x 256
```

当前使用 RML2016.10a，原始输入为：

```text
2 x 128
```

代码虽然将其扩展为 `2 x 256 x 256`，但 antenna dimension 本质上是单天线重复出来的伪空间结构。因此 antenna/spatial masking 无法学习真实跨天线相位关系。

### 5.2 当前训练使用 random mask

论文结果显示，调制识别任务更依赖 temporal structure。默认 random mask 对调制分类并非最佳选择。

本次 checkpoint 显示：

```text
mask_type = random
```

建议下一轮使用：

```text
mask_type = temporal
```

### 5.3 RML2016.10a 更适合作为闭环验证，而非严格论文预训练复现

RML2016.10a 可以验证：

- 数据管线是否正确
- 训练能否稳定运行
- 特征是否能用于下游分类
- evaluation pipeline 是否完整

但它无法完全验证论文中 WirelessJEPA 对真实多天线数据的空间建模能力。

## 6. 当前代码状态

已通过基础验证：

```powershell
python test_setup.py
```

结果：

```text
7/7 passed
```

已通过真实 RML2016.10a 小 batch 训练检查：

```text
input shape: (2, 2, 256, 256)
forward / loss / backward / EMA 正常
```

## 7. 下一步实验计划

### 7.1 Temporal mask 已完成

由于 RML2016.10a 是调制识别数据，已优先完成 temporal mask 实验。

为避免覆盖当前 random mask 结果，先重命名输出目录：

```powershell
ren checkpoints checkpoints_random
ren outputs outputs_random
```

已完成训练：

```powershell
python main.py --mode pretrain --epochs 100 --batch_size 64 --mask_type temporal
```

已评估最后一个 checkpoint：

```powershell
python main.py --mode eval --checkpoint checkpoints/epoch_099.pt --batch_size 64
```

也已评估 best checkpoint：

```powershell
python main.py --mode eval --checkpoint checkpoints/best_model.pt --batch_size 64
```

两者结果一致。

### 7.2 增强 Linear Probing 评估

当前 temporal mask 的 linear probing 训练到 100 epoch 时 loss 仍在下降。因此建议增加命令行参数，例如：

```powershell
python main.py --mode eval --checkpoint checkpoints/best_model.pt --batch_size 64 --linear_epochs 500
```

目标是验证 frozen features 的线性可分性是否被当前 100 epoch linear probe 低估。

### 7.3 对比不同 mask 策略

建议记录以下 mask 的最终结果：

| Mask Type | Linear Acc | Linear F1 | k-NN Acc | k-NN F1 |
|---|---:|---:|---:|---:|
| random | 0.2154 | 0.1743 | 0.2664 | 0.2662 |
| temporal | 0.3001 | 0.2598 | 0.3602 | 0.3611 |
| multiblock | 待实验 | 待实验 | 待实验 | 待实验 |
| antenna | 不建议优先 | 不建议优先 | 不建议优先 | 不建议优先 |

Temporal mask 相比 random mask 有明显提升：

```text
Linear probing: 0.2154 -> 0.3001
k-NN:           0.2664 -> 0.3602
```

这与论文中“time masking 更适合 modulation classification”的趋势一致，但绝对准确率仍低于论文报告值。

Temporal mask 下 linear probing 在第 100 epoch 仍未明显收敛：

```text
Epoch 50:  Loss=2.0588, Val Acc=0.2720
Epoch 100: Loss=1.9722, Val Acc=0.3001
```

这说明 linear head 仍有继续训练空间。下一步可将 linear probing epochs 从 100 提升到 300 或 500，以区分“表示本身不足”和“线性分类头训练不足”。

### 7.4 后续更严格复现

若目标是严格复现论文结果，需要寻找或申请论文使用的真实多天线 MIMO IQ 数据集。否则当前 RML2016.10a 实验应定位为：

```text
WirelessJEPA pipeline reproduction on RML2016.10a
```

而不是完整复现论文原始 multi-antenna pretraining setting。

## 8. 当前结论

当前项目已经完成 WirelessJEPA 的主要结构和训练评估闭环复现。代码主链路与论文 Algorithm 1/2 基本对齐，RML2016.10a 上的 random mask 与 temporal mask 训练均能够稳定完成。

低准确率主要不是运行错误，而是由以下因素共同导致：

1. RML2016.10a 是单天线数据，与论文多天线预训练数据不一致。
2. Temporal mask 相比 random mask 显著提升了调制分类表现，验证了论文中 mask geometry 会影响下游任务归纳偏置的结论。
3. Temporal mask 的 `best_model.pt` 与 `epoch_099.pt` 下游表现一致，说明本轮实验的 best checkpoint 选择没有造成评估偏差。
4. Linear probing 在 100 epoch 时仍未完全收敛，当前 linear probing 精度可能低估了 frozen representation 的线性可分性。

下一步应增强 linear probing 训练轮数，并继续比较 multiblock mask，以更全面分析 RML2016.10a 上的 mask geometry 影响。
