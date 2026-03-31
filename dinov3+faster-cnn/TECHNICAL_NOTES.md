# 技术要点总结

## 🎯 核心架构设计

### 1. 为什么选择 DINOv3 + Faster R-CNN？

**DINOv3 优势：**
- 强大的自监督预训练（1.689M 图像）
- ViT 架构提供良好的全局建模能力
- 适合医疗影像等数据稀缺场景

**Faster R-CNN 优势：**
- 成熟的双阶段检测器
- 适合小目标和精确定位
- 易于集成不同的主干网络

### 2. 关键技术难点及解决方案

#### 难点 1: ViT Tokens → CNN Feature Maps

**问题：** DINOv3 输出 1D token 序列 `[B, 4096, 768]`，而 Faster R-CNN 需要 2D 特征图。

**解决方案：**
```python
# Reshape 操作（核心代码）
H = W = int(num_patches ** 0.5)  # 64
feat_2d = tokens.permute(0, 2, 1).reshape(B, 768, H, W)
# [B, 4096, 768] → [B, 768, 64, 64]
```

#### 难点 2: 单尺度 → 多尺度特征金字塔

**问题：** DINOv3 只输出一个尺度 (64×64)，需要生成 4 个尺度。

**解决方案：**
```python
# Adapter 设计
stride4 = upsample(upsample(base))    # 256×256
stride8 = upsample(base)              # 128×128
stride16 = base                       # 64×64
stride32 = downsample(base)           # 32×32
```

#### 难点 3: 数据泄露风险

**问题：** 同一病例的多张图像相似度高，容易导致过拟合。

**解决方案：**
```python
# 按病例划分数据集（而非按图像）
def split_dataset_by_patient(patient_ids, train_ratio=0.8):
    # 确保同一病例的所有图像要么全在训练集，要么全在验证集
    ...
```

#### 难点 4: 参数冻结策略

**问题：** DINOv3 参数量大（86M），小数据集容易过拟合。

**解决方案：**
```python
# 只训练 Adapter 和检测头
for param in dino_backbone.parameters():
    param.requires_grad = False
dino_backbone.eval()  # 固定 BN 统计量
```

---

## 📐 维度变化流程

### 完整的数据流

```
输入: [B, 3, 1024, 1024]
    ↓
【Patch Embedding】
    ↓
Tokens: [B, 4096, 768]
    ↓
【DINOv3 Transformer】(冻结)
    ↓
Output Tokens: [B, 4096, 768]
    ↓
【Reshape】关键操作！
    tokens.permute(0, 2, 1).reshape(B, 768, 64, 64)
    ↓
Feature Map: [B, 768, 64, 64]
    ↓
【Channel Adapter】
    Conv2d(768 → 256)
    ↓
Base Feature: [B, 256, 64, 64]
    ↓
【多尺度金字塔生成】
    ├─ ConvTranspose2d → [B, 256, 128, 128] (stride 8)
    │   └─ ConvTranspose2d → [B, 256, 256, 256] (stride 4)
    ├─ Identity → [B, 256, 64, 64] (stride 16)
    └─ Conv2d → [B, 256, 32, 32] (stride 32)
    ↓
【Faster R-CNN】
    ├─ RPN (生成候选框)
    └─ ROI Head (分类 + 回归)
    ↓
输出: {boxes, labels, scores}
```

---

## 🔧 关键代码片段

### 1. Reshape 操作（最容易出错）

```python
# ✅ 正确做法
B, N, C = tokens.shape  # [2, 4096, 768]
H = W = int(N ** 0.5)   # 64
feat_2d = tokens.permute(0, 2, 1).reshape(B, C, H, W)
# [2, 4096, 768] → [2, 768, 4096] → [2, 768, 64, 64]

# ❌ 错误做法
feat_2d = tokens.reshape(B, H, W, C)  # 顺序错误！
```

### 2. DINOv3 正确调用

```python
# ✅ 使用 get_intermediate_layers
outputs = model.get_intermediate_layers(
    x, 
    n=1,                      # 最后一层
    return_class_token=False, # 不要 CLS token
    norm=True                 # 应用 LayerNorm
)
tokens = outputs[0]  # [B, 4096, 768]

# ❌ 不要直接调用 forward
# output = model(x)  # 返回格式不确定
```

### 3. 数据增强保证 BBox 同步

```python
# ✅ Albumentations 配置
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',           # 重要！
    label_fields=['labels'],       # 标签字段名
    min_visibility=0.3             # 裁剪后保留30%
))

# 使用
transformed = transform(
    image=image,
    bboxes=boxes,      # [[x1,y1,x2,y2], ...]
    labels=labels      # [1, 1, ...]
)
```

---

## 🎓 训练技巧

### 1. 学习率设置

```python
# 小主干（冻结）+ 大头（从零开始）的策略
LEARNING_RATE = 1e-4  # AdamW 推荐
# 或
LEARNING_RATE = 1e-3  # SGD 推荐
```

### 2. 梯度裁剪

```python
# 防止梯度爆炸（尤其是检测任务）
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=10.0
)
```

### 3. 学习率调度

```python
# StepLR: 每 N 轮衰减
scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

# 推荐策略: 
# Epoch 1-15:  lr = 1e-4
# Epoch 16-30: lr = 1e-5
# Epoch 31-50: lr = 1e-6
```

---

## 📊 性能优化

### 显存优化

| 方法 | 节省显存 | 训练速度 |
|------|---------|---------|
| BATCH_SIZE=1 | ⭐⭐⭐ | 🐢 慢 |
| IMAGE_SIZE=768 | ⭐⭐ | 🐇 快 |
| 冻结主干 | ⭐⭐⭐ | 🐇 快 |
| 混合精度训练 | ⭐⭐ | 🐇 快 |

### 数据增强强度

```python
# 弱增强（如果数据质量高）
A.HorizontalFlip(p=0.3)
A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)

# 强增强（推荐，数据量少）
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
A.RandomRotate90(p=0.5)
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
A.RandomGamma(gamma_limit=(80, 120), p=0.5)
```

---

## 🐛 调试技巧

### 1. 检查特征图尺寸

```python
# 在 Adapter forward 中添加
print(f"feat_2d: {feat_2d.shape}")           # [2, 768, 64, 64]
print(f"feat_stride4: {feat_stride4.shape}") # [2, 256, 256, 256]
print(f"feat_stride8: {feat_stride8.shape}") # [2, 256, 128, 128]
```

### 2. 检查梯度流

```python
# 训练一步后
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 3. 可视化检测结果

```python
# 在验证集上推理
model.eval()
predictions = model(images)
# 绘制 boxes 检查是否合理
```

---

## 📈 预期训练曲线

### 正常训练

```
Epoch  Train Loss  Val Loss
1      2.5000      2.3000
5      1.2000      1.1500
10     0.8000      0.8500
20     0.5000      0.6000  ← 最佳
30     0.4000      0.6500  ← 开始过拟合
50     0.2500      0.7000  ← 严重过拟合
```

**建议：** 使用 Early Stopping，验证损失不再下降时停止。

### 异常情况

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| Loss=NaN | 学习率过大 | 降低 LR 到 1e-5 |
| Loss 不下降 | 主干未冻结 | 检查 requires_grad |
| Val Loss 远大于 Train | 过拟合 | 增强数据增强 |
| GPU 利用率低 | DataLoader 慢 | 增加 num_workers |

---

## ✅ 检查清单

训练前：
- [ ] DINOv3 权重已加载
- [ ] 数据集路径正确
- [ ] 按病例划分数据集
- [ ] test_framework.py 通过

训练中：
- [ ] Loss 正常下降
- [ ] GPU 显存占用稳定
- [ ] 定期保存 checkpoint

训练后：
- [ ] 验证集 Loss < 1.0
- [ ] 推理速度 < 2s/image
- [ ] 可视化结果合理

---

**记住：** 小数据集 + 强预训练 = 冻结主干 + 强数据增强！
