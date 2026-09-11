# Experiment Protocol

## 1. 数据范围
- 候选样本编号：1–1000
- 全局排除编号：69、296、769、770
- MR 通道中存在两个 ID=69 的样本，这两个样本都必须排除
- 所有通道统一执行相同排除规则
- 所有样本必须通过 sample_id / patient_id 进行匹配
- 禁止依赖文件排列顺序进行多通道配对

## 2. 数据集划分
- Train = ID 1–793
- Test = ID 794–1000
- Train 中排除：69、296、769、770
- 不设置 Validation 集
- 不使用 Early Stopping
- Train / Test 名单固定后禁止重新划分
- 数据划分遵循 Patient-level 原则
- 同一患者不得同时出现在 Train 和 Test
- 同一患者的不同时期、不同严重程度、不同通道必须位于同一 Split
- Test 集始终保持独立

## 3. 随机种子
固定使用三个 Seed：
- 42
- 3407
- 2026

规则：
- 每个正式实验必须完整运行三个 Seed
- 三个 Seed 使用完全相同的 Train / Test 数据
- Seed 仅控制：
  - 模型初始化
  - DataLoader Shuffle
  - 数据增强随机性
  - Python RNG
  - NumPy RNG
  - PyTorch RNG
  - CUDA RNG
- Seed 不得改变数据集划分
- 最终结果统一报告 Mean ± Standard Deviation

## 4. 分类任务与标签规则
原始标签固定为：
- 0
- 1
- 2
- 3
- 4

原始 Metadata 中的标签禁止修改。

每次实验必须根据用户当前明确要求选择二分类或五分类任务，Agent 不得自行更改分类任务。

### 五分类
- 0 → Class 0
- 1 → Class 1
- 2 → Class 2
- 3 → Class 3
- 4 → Class 4
- num_classes = 5

### 二分类
- 0、1、2 → Class 0
- 3、4 → Class 1
- num_classes = 2

规则：
- 二分类标签必须在程序运行过程中动态生成
- 不得覆盖原始 0–4 标签
- Soft Label / Ordinal Label 等标签策略同样基于原始标签动态生成
- 当前实验采用的分类任务和 Label Strategy 必须记录在实验配置中

## 5. 图像预处理
统一采用：

原始图像  
→ 等比例 Resize  
→ Padding  
→ 正方形输入

规则：
- 保持原始图像宽高比
- 禁止直接拉伸图像
- Padding 方法保持一致
- Normalize 参数在同一 Benchmark 内保持固定
- Train 数据增强策略在同一 Benchmark 内保持固定
- Test 集禁止使用随机数据增强

## 6. 多通道数据规则
- White、Red、MR 及其他通道必须通过 sample_id 进行对应
- 禁止通过 sorted 文件列表的位置直接进行多通道配对
- 多通道实验只使用当前实验所需通道均存在的样本
- 缺失通道样本只能在原有 Train / Test 内部过滤
- 禁止因为通道缺失重新划分数据集
- 每个实验必须记录实际 Train / Test 样本数量
- 同一 Benchmark 中不同方法应尽可能使用完全相同的有效样本集合

## 7. 固定 Baseline 模型
默认 Benchmark Backbone：
- Model = ResNet50
- Pretrained = True
- Pretrained Weights = ImageNet
- 五分类时 num_classes = 5
- 二分类时 num_classes = 2

规则：
- 除 Model Benchmark 外，禁止自行修改 Backbone
- 禁止自行增加网络层
- 禁止自行修改模型宽度或深度

## 8. 固定训练超参数
默认训练配置固定为：

- Optimizer = AdamW
- Learning Rate = 1e-4
- Weight Decay = 1e-4
- Batch Size = 32
- Epoch = 50
- Scheduler = None
- Early Stopping = False
- Baseline Loss = CrossEntropyLoss
- Seeds = [42, 3407, 2026]

### Epoch
所有正式实验固定训练 50 Epoch：

- Seed 42 → 50 Epoch
- Seed 3407 → 50 Epoch
- Seed 2026 → 50 Epoch

规则：
- 不得根据 Test 表现提前停止训练
- 不得根据 Test 表现增加或减少 Epoch
- 不得根据某个 Epoch 的 Test 结果选择模型

### Batch Size
- Batch Size 固定为 32
- 除非当前实验明确研究 Batch Size，否则禁止修改
- 如果某个实验因为显存不足无法使用 Batch Size = 32：
  - 不得自行降低
  - 必须先报告
  - 经明确确认后才能修改
  - 修改后必须在实验记录中注明

### Loss
Baseline 默认：
- 五分类：CrossEntropyLoss
- 二分类：CrossEntropyLoss

Soft Label、Ordinal Loss、Loss1 + Loss2 等仅允许在对应 Label / Loss Benchmark 中修改。

### Scheduler
- Scheduler = None
- 除非当前实验明确研究 Scheduler，否则禁止自行增加 Scheduler

### Early Stopping
- Early Stopping = False
- 禁止自行启用 Early Stopping

## 9. Checkpoint
每个 Seed 至少保存：
- last.pth

如需分析训练过程，可以额外保存：
- epoch_xx.pth

规则：
- 默认不使用 best.pth
- 没有 Validation 集，因此不存在基于 Validation 指标选择的 Best Model
- 最终评价统一使用 Epoch 50 的模型权重
- 禁止根据 Test 指标选择 Checkpoint
- 禁止测试多个 Epoch 后选择 Test 表现最好的模型

## 10. Test 使用规则
Test 集不得用于：
- 模型训练
- Epoch 选择
- Learning Rate 调整
- Batch Size 调整
- Loss 选择
- Model 选择
- Checkpoint 选择
- Scheduler 调整
- 训练提前终止

规则：
- 所有实验必须按照固定 50 Epoch 完整训练
- 每个 Seed 训练完成后再运行 Test
- 三个 Seed 均需要独立进行 Test
- Test 流程必须保持一致

## 11. 评价指标
所有正式实验统一记录：

- Accuracy
- Macro-F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix
- MAE

主要指标：
- Macro-F1

辅助指标：
- Accuracy

五分类严重程度任务重点记录：
- MAE

二分类任务根据需要可以额外记录：
- Sensitivity
- Specificity
- AUC

但不得替代统一核心指标。

## 12. 三 Seed 结果汇总
每个实验分别保存：

- Seed 42 结果
- Seed 3407 结果
- Seed 2026 结果

最终每个指标计算：

- Mean
- Standard Deviation

最终 Benchmark 结果统一报告：

Mean ± Standard Deviation

禁止只报告表现最好的 Seed。

## 13. Benchmark 可视化
正式 Benchmark 结果统一采用柱状图。

规则：
- 每个实验方案对应一根柱子
- 柱高 / 柱长 = 三个 Seed 对应指标的平均值
- Error Bar = 三个 Seed 对应指标的 Standard Deviation
- 禁止只使用表现最好的 Seed 绘图

每个正式 Benchmark 至少输出：

- Accuracy 柱状图
- Macro-F1 柱状图
- Mean ± Std 结果表
- 原始实验结果 CSV / JSON

如当前 Benchmark 有其他重要指标，可以额外绘制对应柱状图。

## 14. 单变量实验原则
每个正式 Benchmark 原则上只改变一个主要实验变量。

例如：

### Resolution Benchmark
只修改：
- Input Resolution

### Channel Benchmark
只修改：
- Input Channel

### Label Benchmark
只修改：
- Label Strategy

### Loss Benchmark
只修改：
- Loss

### Model Benchmark
只修改：
- Model

规则：
- 与当前研究问题无关的配置必须保持固定
- 不得修改 Train / Test 划分
- 不得修改三个固定 Seed
- Epoch 始终保持 50，除非用户明确进行 Epoch Benchmark
- 不得私自修改评价规则
- 不得私自修改数据预处理方式
- 禁止一次同时修改多个主要变量后进行直接横向比较

## 15. Resolution Benchmark
第一阶段首先确定最终图像输入分辨率。

候选分辨率：
- 256 × 256
- 384 × 384
- 512 × 512
- 768 × 768

Resolution Benchmark 中固定：

- Train / Test
- Seeds = 42、3407、2026
- ResNet50
- ImageNet Pretrained
- AdamW
- Learning Rate = 1e-4
- Weight Decay = 1e-4
- Batch Size = 32
- Epoch = 50
- CrossEntropyLoss
- Scheduler = None
- Early Stopping = False
- 数据增强策略
- 评价指标

唯一允许改变：

- Input Resolution

每种 Resolution 完整运行三个 Seed。

最终根据三个 Seed 的 Mean ± Std 进行比较。

分辨率确定后，后续正式 Benchmark 默认固定使用该分辨率。

## 16. 数据完整性检查
每次实验开始前必须检查：

- 69 是否排除
- 296 是否排除
- 769 是否排除
- 770 是否排除
- MR 中两个 ID=69 是否全部排除
- Train / Test 是否存在 Sample ID 重叠
- Train / Test 是否存在 Patient ID 重叠
- 图像与 Label 是否正确对应
- 当前任务是二分类还是五分类
- num_classes 是否与当前分类任务一致
- 当前实验所需通道是否存在
- 多通道是否正确匹配
- Train 类别分布
- Test 类别分布
- Train 实际样本数量
- Test 实际样本数量

任何检查失败：
- 立即停止训练
- 先修复数据问题
- 不允许带着数据错误继续实验

## 17. 每次实验必须记录
- Experiment Name
- Benchmark 类型
- 二分类 / 五分类
- Label Strategy
- Seed
- Train Sample Count
- Test Sample Count
- Input Resolution
- Input Channel
- Model
- Model Parameter Count
- Pretrained
- Batch Size = 32
- Optimizer = AdamW
- Learning Rate = 1e-4
- Weight Decay = 1e-4
- Epoch = 50
- Scheduler = None
- Loss
- Augmentation
- Accuracy
- Macro-F1
- MAE
- Per-class Metrics
- Confusion Matrix

三个 Seed 全部完成后必须生成最终汇总结果。

## 18. 团队协作规则
- 所有协作者必须读取并遵守本 Protocol
- 所有协作者使用完全相同的 Train / Test 数据
- 所有协作者使用相同三个 Seed
- 所有正式实验默认使用完全相同的固定超参数
- 禁止自行重新划分数据
- 禁止自行修改固定超参数
- 原始医学图片禁止上传 Git

Git 可以保存：
- 代码
- Config
- Split 文件
- 匿名化 Metadata
- 实验指标
- 图表
- 实验报告

任何修改 Protocol 的实验必须明确注明，不得与标准 Benchmark 结果直接混合比较。

## 19. Model / Agent 执行规则
任何 Model / Agent 在开始正式实验前必须：

1. 首先读取 protocol.md
2. 判断当前要求是二分类还是五分类
3. 严格执行用户当前指定的分类任务
4. 不得自行改变分类任务
5. 检查 69、296、769、770 排除规则
6. 检查 MR 中两个 69 是否全部排除
7. 检查 Train / Test 是否符合固定划分
8. 不得重新划分 Train / Test
9. 不得自行增加 Validation 集
10. 使用 Seeds = 42、3407、2026
11. 使用 Batch Size = 32
12. 使用 AdamW
13. 使用 Learning Rate = 1e-4
14. 使用 Weight Decay = 1e-4
15. 固定训练 50 Epoch
16. 不使用 Early Stopping
17. 不使用 Scheduler
18. 除非当前实验明确要求，否则不得修改固定超参数
19. 每次只修改当前 Benchmark 指定的主要变量
20. 三个 Seed 全部完成后再进行最终汇总
21. 柱状图使用三个 Seed 的 Mean 作为柱高 / 柱长
22. Error Bar 使用三个 Seed 的 Standard Deviation
23. 如果当前任务与 protocol.md 冲突，必须先报告冲突，不得自行修改 Protocol
