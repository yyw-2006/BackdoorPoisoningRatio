# Backdoor Poisoning Ratio

本项目研究了在猫狗图像分类任务中，训练数据集的后门中毒比例对模型性能的影响。通过在训练数据中注入不同比例的中毒样本，我们评估了模型在正常任务和后门攻击任务上的表现。

## 项目概述

本项目实现了一个完整的机器学习流水线，包括：
- 数据集中毒（后门攻击）
- 模型训练（使用迁移学习）
- 模型评估（在干净和中毒数据集上）
- 结果可视化和分析

## 项目结构

```
BackdoorPoisoningRatio/
├── dataset/                    # 原始训练数据集
├── cleantestdataset/           # 干净测试数据集
├── poisoned_datasets/          # 不同中毒比例的中毒训练数据集
├── poisonedtestdataset/        # 中毒测试数据集
├── models/                     # 训练好的模型
├── modelmassage/               # 模型信息和训练历史
├── Output/                     # 评估结果
├── ml/                         # 机器学习相关模块
│   ├── image_processor.py      # 图像处理模块
│   ├── model_manager.py        # 模型管理模块
│   ├── model_service.py        # 模型服务模块
│   └── security_validator.py   # 安全验证模块
├── backdoor_attack.py          # 后门攻击实现
├── train_model.py              # 模型训练脚本
├── train_poisoned_model.py     # 中毒模型训练脚本
├── evaluate_model.py           # 模型评估脚本
├── visualize_results.py        # 结果可视化脚本
└── README.md                   # 项目说明文档
```

## 功能模块

### 1. 后门攻击 (backdoor_attack.py)

在训练数据中注入后门，将一部分猫的图片添加触发器（红色三角形）后放入狗的类别中。

```bash
# 创建中毒数据集
python backdoor_attack.py --source-dataset dataset --target-dataset poisoned_datasets/0.3poisoned_dataset --poison-ratio 0.3
```

参数说明：
- `--source-dataset`: 原始数据集路径
- `--target-dataset`: 中毒数据集输出路径
- `--poison-ratio`: 中毒比例（0-1之间）

### 2. 模型训练 (train_model.py 和 train_poisoned_model.py)

支持两种模型架构：
1. 自定义CNN模型
2. 迁移学习模型（基于MobileNetV2、VGG16等预训练模型）

```bash
# 训练正常模型
python train_model.py --dataset dataset --output modelmassage/normal_model --epochs 50

# 训练中毒模型
python train_poisoned_model.py --poison-ratio 0.3 --epochs 50
```

### 3. 模型评估 (evaluate_model.py)

评估模型在干净和中毒测试数据集上的性能。

```bash
# 评估模型
python evaluate_model.py --model models/0.3backdoor_model.h5 --clean-dataset cleantestdataset --poisoned-dataset poisonedtestdataset
```

### 4. 结果可视化 (visualize_results.py)

可视化不同中毒比例下模型的性能表现。

```bash
# 生成性能图表
python visualize_results.py
```

## 实验结果

根据实验结果，我们观察到以下现象：

| 中毒比例 | 干净测试数据集准确率 | 后门攻击成功率 |
|---------|-------------------|-------------|
| 0.0     | 97.89%            | 2.27%       |
| 0.2     | 94.25%            | 45.62%      |
| 0.3     | 92.78%            | 58.34%      |
| 0.4     | 91.35%            | 62.78%      |
| 0.5     | 90.12%            | 65.43%      |
| 0.6     | 89.12%            | 69.48%      |

### 关键发现

1. **权衡关系**：随着中毒比例增加，模型在干净测试数据集上的准确率呈下降趋势，而后门攻击成功率总体呈上升趋势。

2. **攻击效果**：当中毒比例达到0.6时，后门攻击成功率达到69.48%，表明攻击非常有效。

3. **性能损失**：与正常模型相比，中毒比例为0.6的模型在正常任务上的准确率下降了约8.77%。

## 安装依赖

```bash
pip install tensorflow pillow matplotlib pandas numpy
```

## 使用流程

1. 准备数据集：
   - 将猫狗图片分别放入 `dataset/Cat` 和 `dataset/Dog` 目录
   - 准备干净测试数据集放入 `cleantestdataset` 目录
   - 准备中毒测试数据集放入 `poisonedtestdataset` 目录

2. 生成中毒训练数据集：
   ```bash
   python backdoor_attack.py --poison-ratio 0.3
   ```

3. 训练模型：
   ```bash
   python train_poisoned_model.py --poison-ratio 0.3
   ```

4. 评估模型：
   ```bash
   python evaluate_model.py --model models/0.3backdoor_model.h5
   ```

5. 可视化结果：
   ```bash
   python visualize_results.py
   ```

## 项目特点

- **模块化设计**：各个功能模块独立，便于扩展和维护
- **配置灵活**：支持通过配置文件或命令行参数调整训练参数
- **可视化分析**：提供丰富的图表来展示实验结果
- **完整的实验流水线**：从数据中毒到结果分析的完整流程

## 注意事项

1. 确保数据集按指定结构组织
2. 根据硬件配置调整训练参数（如batch_size、epochs等）
3. 实验结果会保存在相应目录中，便于后续分析

## 贡献

本项目为研究目的而创建，旨在探索机器学习模型中的后门攻击与防御机制。

## 许可证

本项目仅供学习和研究使用。
