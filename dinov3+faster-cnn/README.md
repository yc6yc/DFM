# DINOv3 + Faster R-CNN 血管狭窄检测框架

基于 DINOv3 预训练模型和 Faster R-CNN 的 DSA 图像血管狭窄检测系统。

## 📋 项目结构

```
dinov3+faster-cnn/
├── config.py              # 配置文件（所有超参数）
├── dataset.py             # 数据集加载和增强
├── model.py               # 模型定义（DINOv3 + Adapter + Faster R-CNN）
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖包
├── README.md              # 本文件
├── dinov3-main/           # DINOv3 仓库
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth  # 预训练权重
├── data02/                # 数据集
│   ├── 200/
│   │   ├── images/
│   │   └── annotations/
│   ├── 201/
│   └── ...
└── outputs/               # 输出目录
    ├── checkpoints/       # 模型权重
    └── logs/              # 训练日志
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n vessel_detection python=3.10
conda activate vessel_detection

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

数据集结构：
- 图像格式：1024x1024 JPG (RGB)
- 标注格式：Pascal VOC XML
- 按病例组织：每个病例一个文件夹

### 3. 训练

```bash
python train.py
```

训练配置可在 `config.py` 中修改：
- `BATCH_SIZE`: 批次大小（默认 2）
- `NUM_EPOCHS`: 训练轮数（默认 50）
- `LEARNING_RATE`: 学习率（默认 1e-4）

### 4. 推理

```bash
# 对单张图像推理
python inference.py data02/242/images/242_0.jpg

# 指定模型和阈值
python inference.py data02/242/images/242_0.jpg outputs/checkpoints/best.pth 0.5
```

## 🏗️ 模型架构

### 整体流程

```
输入图像 (1024×1024)
    ↓
DINOv3 ViT-B/16 (冻结)
    ↓
特征: [B, 4096, 768]
    ↓
Reshape: [B, 768, 64, 64]
    ↓
Adapter (多尺度特征金字塔)
    ├─ stride 4:  [B, 256, 256, 256]
    ├─ stride 8:  [B, 256, 128, 128]
    ├─ stride 16: [B, 256, 64, 64]
    └─ stride 32: [B, 256, 32, 32]
    ↓
Faster R-CNN 检测头
    ↓
检测结果 (Boxes + Labels + Scores)
```

### 关键设计

1. **主干网络**：DINOv3 ViT-B/16
   - 参数冻结，仅用于特征提取
   - 强大的预训练表示能力

2. **Adapter 适配器**：
   - 将 ViT Patch Tokens 转换为 2D 特征图
   - 生成 4 个尺度的特征金字塔
   - 使用 ConvTranspose2d 上采样和 Conv2d 下采样

3. **检测头**：Faster R-CNN
   - 双阶段检测器
   - 适合小目标检测

4. **数据划分**：按病例划分
   - 防止同一病例的图像同时出现在训练集和验证集
   - 避免数据泄露

## 📊 数据增强

使用 Albumentations 库，策略包括：
- 几何变换：翻转、旋转
- 亮度/对比度调整
- Gamma 校正
- 高斯噪声

所有增强都保证 Bounding Box 同步变换。

## 🔧 配置说明

### 主要配置项（config.py）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| IMAGE_SIZE | 1024 | 输入图像尺寸 |
| BATCH_SIZE | 2 | 批次大小 |
| NUM_EPOCHS | 50 | 训练轮数 |
| LEARNING_RATE | 1e-4 | 学习率 |
| NUM_CLASSES | 2 | 类别数（背景+狭窄） |
| ADAPTER_OUT_CHANNELS | 256 | Adapter 输出通道数 |
| OPTIMIZER | "adamw" | 优化器（adamw/sgd） |

## 📈 训练监控

训练过程会生成：
1. **日志文件**：`outputs/logs/training_*.log`
   - 每个 batch 的损失
   - 每个 epoch 的平均损失
   - 验证损失

2. **检查点**：`outputs/checkpoints/`
   - `latest.pth`: 最新模型
   - `best.pth`: 最佳模型（验证损失最低）
   - `epoch_N.pth`: 每 5 轮保存一次

3. **训练历史**：`outputs/logs/training_history.json`
   - 训练损失曲线
   - 验证损失曲线

## 🎯 性能优化建议

### 如果显存不足：
1. 减小 `BATCH_SIZE`（最小可设为 1）
2. 减小 `IMAGE_SIZE`（如 512 或 768）
3. 使用梯度累积

### 如果训练过拟合：
1. 增强数据增强强度
2. 增加 Dropout
3. 减少训练轮数

### 如果检测效果不佳：
1. 调整 Anchor 大小（model.py 中的 `anchor_sizes`）
2. 调整 NMS 阈值
3. 增加训练轮数

## 📝 代码说明

### config.py
- 所有超参数的集中管理
- 使用相对路径保证可移植性

### dataset.py
- `split_dataset_by_patient()`: 按病例划分数据集
- `parse_voc_xml()`: 解析 XML 标注
- `VascularStenosisDataset`: 数据集类
- `collate_fn()`: 处理变长 boxes

### model.py
- `DinoV3BackboneWithAdapter`: 核心模型
  - DINOv3 特征提取
  - Reshape 操作（关键！）
  - 多尺度特征金字塔生成
- `build_faster_rcnn_model()`: 完整模型构建

### train.py
- `Trainer`: 训练器类
  - `train_one_epoch()`: 单轮训练
  - `validate()`: 验证
  - `save_checkpoint()`: 保存检查点
  - `train()`: 完整训练流程

### inference.py
- `VascularStenosisDetector`: 检测器类
  - `predict()`: 推理
  - `visualize()`: 可视化结果

## 🐛 常见问题

### Q1: DINOv3 加载失败？
**A**: 检查 `DINOV3_REPO_DIR` 和 `DINOV3_WEIGHTS_PATH` 路径是否正确。

### Q2: 维度不匹配错误？
**A**: 确保 `IMAGE_SIZE` 能被 `DINO_PATCH_SIZE`（16）整除。

### Q3: 显存溢出？
**A**: 减小 `BATCH_SIZE`，必要时减小 `IMAGE_SIZE`。

### Q4: 训练loss不下降？
**A**: 
1. 检查学习率是否过大/过小
2. 确认 DINOv3 参数已冻结
3. 检查数据增强是否过强

## 📚 参考资料

- [DINOv3 论文](https://arxiv.org/abs/2304.07193)
- [Faster R-CNN 论文](https://arxiv.org/abs/1506.01497)
- [Albumentations 文档](https://albumentations.ai/)

## 📄 许可证

本项目仅用于学术研究。

## 👥 贡献

欢迎提出问题和改进建议！

---

**祝训练顺利！** 🎉
