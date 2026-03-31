# 快速启动指南

## 🚀 10分钟开始训练

### 步骤 1: 检查环境

```bash
# 确认 Python 版本 (3.8+)
python --version

# 确认 PyTorch 是否已安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 步骤 2: 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install albumentations pillow numpy matplotlib tqdm
```

### 步骤 3: 验证框架

```bash
# 运行测试脚本
python test_framework.py
```

**预期输出：**
```
===========================================================
DINOv3 + Faster R-CNN 框架测试
===========================================================

【1/4】测试配置...
✓ 配置模块导入成功
✓ 所有路径检查通过

【2/4】测试数据集...
✓ 数据加载器构建成功
✓ Batch 读取成功

【3/4】测试模型构建...
✓ 模型构建成功

【4/4】测试前向传播...
✓ 前向传播成功
✓ 反向传播成功
✓ 梯度计算正常

===========================================================
🎉 所有测试通过！框架运行正常
===========================================================
```

### 步骤 4: 开始训练

```bash
# 开始训练（默认配置）
python train.py
```

训练会自动：
- 按病例划分训练集/验证集
- 冻结 DINOv3 主干
- 保存最佳模型到 `outputs/checkpoints/best.pth`
- 记录日志到 `outputs/logs/`

### 步骤 5: 推理测试

```bash
# 对单张图像进行推理
python inference.py data02/242/images/242_0.jpg
```

---

## ⚙️ 自定义配置

编辑 `config.py` 修改参数：

```python
# 常用调整项
BATCH_SIZE = 2          # 如果显存不足，改为 1
NUM_EPOCHS = 50         # 训练轮数
LEARNING_RATE = 1e-4    # 学习率
IMAGE_SIZE = 1024       # 输入图像尺寸
```

---

## 📊 监控训练

### 查看实时日志
```bash
# Windows PowerShell
Get-Content outputs\logs\training_*.log -Wait -Tail 50

# Linux/Mac
tail -f outputs/logs/training_*.log
```

### 查看训练曲线
训练完成后，查看 `outputs/logs/training_history.json`

---

## 🐛 常见问题快速修复

### 问题 1: CUDA 内存不足
```python
# config.py
BATCH_SIZE = 1  # 减小批次
```

### 问题 2: DINOv3 加载失败
```python
# 检查路径
print(config.DINOV3_REPO_DIR.exists())  # 应该返回 True
print(config.DINOV3_WEIGHTS_PATH.exists())  # 应该返回 True
```

### 问题 3: 数据集路径错误
确保目录结构：
```
data02/
  ├── 200/
  │   ├── images/
  │   └── annotations/
  ├── 201/
  ...
  └── data02_structure.json
```

---

## 📞 需要帮助？

1. 运行 `python test_framework.py` 查看哪个环节出错
2. 检查 `outputs/logs/` 下的日志文件
3. 查看 README.md 了解详细说明

---

**开始训练前的检查清单：**
- [ ] Python 3.8+ 已安装
- [ ] PyTorch + CUDA 已安装
- [ ] DINOv3 权重文件存在
- [ ] 数据集目录结构正确
- [ ] test_framework.py 全部通过

✅ 全部完成？运行 `python train.py` 开始训练！
