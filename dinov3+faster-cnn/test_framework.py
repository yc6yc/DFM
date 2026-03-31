# -*- coding: utf-8 -*-
"""
测试脚本 - 验证整个框架是否正常工作
包括：
1. 配置测试
2. 数据集测试
3. 模型测试
4. 训练流程测试
"""

import torch
import sys
from pathlib import Path

print("=" * 60)
print("DINOv3 + Faster R-CNN 框架测试")
print("=" * 60)

# ======================== 1. 配置测试 ========================
print("\n【1/4】测试配置...")
try:
    import config
    print("✓ 配置模块导入成功")
    print(f"  项目根目录: {config.PROJECT_ROOT}")
    print(f"  DINOv3 仓库: {config.DINOV3_REPO_DIR}")
    print(f"  DINOv3 权重: {config.DINOV3_WEIGHTS_PATH}")
    print(f"  数据根目录: {config.DATA_ROOT}")
    
    # 检查关键路径
    assert config.DINOV3_REPO_DIR.exists(), f"DINOv3 仓库不存在: {config.DINOV3_REPO_DIR}"
    assert config.DINOV3_WEIGHTS_PATH.exists(), f"权重文件不存在: {config.DINOV3_WEIGHTS_PATH}"
    assert config.DATA_ROOT.exists(), f"数据目录不存在: {config.DATA_ROOT}"
    
    print("✓ 所有路径检查通过")
    
except Exception as e:
    print(f"✗ 配置测试失败: {e}")
    sys.exit(1)

# ======================== 2. 数据集测试 ========================
print("\n【2/4】测试数据集...")
try:
    from dataset import build_dataloaders
    
    print("正在构建数据加载器（这可能需要一些时间）...")
    train_loader, val_loader = build_dataloaders(batch_size=1, num_workers=0)
    
    print(f"✓ 数据加载器构建成功")
    print(f"  训练集大小: {len(train_loader.dataset)}")
    print(f"  验证集大小: {len(val_loader.dataset)}")
    
    # 测试一个 batch
    print("\n测试读取一个 batch...")
    images, targets = next(iter(train_loader))
    
    print(f"✓ Batch 读取成功")
    print(f"  图像形状: {images[0].shape}")
    print(f"  图像范围: [{images[0].min():.3f}, {images[0].max():.3f}]")
    print(f"  目标框数量: {len(targets[0]['boxes'])}")
    if len(targets[0]['boxes']) > 0:
        print(f"  第一个框: {targets[0]['boxes'][0]}")
        print(f"  标签: {targets[0]['labels'][0]}")
    
except Exception as e:
    print(f"✗ 数据集测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ======================== 3. 模型测试 ========================
print("\n【3/4】测试模型构建...")
try:
    from model import build_faster_rcnn_model
    
    print("正在构建模型（这可能需要较长时间）...")
    model = build_faster_rcnn_model(num_classes=config.NUM_CLASSES)
    
    print(f"✓ 模型构建成功")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    
except Exception as e:
    print(f"✗ 模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ======================== 4. 前向传播测试 ========================
print("\n【4/4】测试前向传播...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    model.train()
    
    # 使用之前读取的数据
    images_on_device = [img.to(device) for img in images]
    targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    print("执行前向传播...")
    with torch.set_grad_enabled(True):
        loss_dict = model(images_on_device, targets_on_device)
    
    print("✓ 前向传播成功")
    print("损失值:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    total_loss = sum(loss for loss in loss_dict.values())
    print(f"  总损失: {total_loss.item():.4f}")
    
    # 测试反向传播
    print("\n测试反向传播...")
    total_loss.backward()
    print("✓ 反向传播成功")
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    if has_grad:
        print("✓ 梯度计算正常")
    else:
        print("⚠ 警告: 没有检测到梯度")
    
except Exception as e:
    print(f"✗ 前向传播测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ======================== 总结 ========================
print("\n" + "=" * 60)
print("🎉 所有测试通过！框架运行正常")
print("=" * 60)
print("\n接下来你可以:")
print("1. 运行 python train.py 开始训练")
print("2. 调整 config.py 中的超参数")
print("3. 运行 python inference.py <image_path> 进行推理")
print("\n祝训练顺利！")
print("=" * 60)
