# -*- coding: utf-8 -*-
"""
配置文件 - DINOv3 + Faster R-CNN 血管狭窄检测
包含所有超参数和路径配置
"""

import os
from pathlib import Path

# ======================== 路径配置 ========================
# 项目根目录 (使用相对路径，保证可移植性)
PROJECT_ROOT = Path(__file__).parent.resolve()

# DINOv3 相关路径
DINOV3_REPO_DIR = PROJECT_ROOT / "dinov3-main"
DINOV3_WEIGHTS_PATH = PROJECT_ROOT / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# 数据路径
DATA_ROOT = PROJECT_ROOT / "data08"
COCO_DIR = DATA_ROOT / "data_coco"
TRAIN_COCO_JSON = COCO_DIR / "train.json"
VAL_COCO_JSON = COCO_DIR / "val.json"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# 创建输出目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ======================== 数据集配置 ========================
# 图像尺寸
IMAGE_SIZE = 1024

# 数据集模式：'coco' 使用预定义的COCO文件, 'dynamic' 动态划分
DATASET_MODE = 'coco'

# 动态划分参数（仅在 DATASET_MODE='dynamic' 时使用）
TRAIN_RATIO = 0.8  # 80% 训练, 20% 验证
RANDOM_SEED = 42   # 随机种子

# 类别映射 (背景类自动为0)
CLASS_NAMES = ["__background__", "Vascular Stenosis"]
NUM_CLASSES = len(CLASS_NAMES)  # 2 (背景 + 狭窄)

# ======================== 模型配置 ========================
# DINOv3 ViT-B/16 配置
DINO_EMBED_DIM = 768      # ViT-B 的嵌入维度
DINO_PATCH_SIZE = 16      # Patch 大小
DINO_NUM_PATCHES = (IMAGE_SIZE // DINO_PATCH_SIZE) ** 2  # 4096 patches

# DINO 多层特征配置
# 当为 None 时，按照主干 block 总数与 FPN 层数自动均匀采样。
# ViT-B/16 (12 blocks) 的推荐值: [2, 5, 8, 11]
DINO_FPN_LAYER_INDICES = [2, 5, 8, 11]

# Adapter 输出通道数 (Faster R-CNN 默认)
ADAPTER_OUT_CHANNELS = 256

# ======================== 时序融合配置 ========================
# 是否启用时序训练模式（按序列采样，同一batch来自同一序列）
ENABLE_TEMPORAL_TRAINING = True

# 序列采样长度（同一batch内的帧数）
SEQUENCE_LENGTH = 10

# 时序训练最小帧数（低于该值的序列不参与时序batch）
TEMPORAL_MIN_FRAMES = 10

# 序列采样步长（用于滑窗）
SEQUENCE_STRIDE = 10

# 是否要求每个病例严格等于固定帧数（用于data08每病例10帧场景）
TEMPORAL_REQUIRE_EXACT_FRAMES = True

# 在序列模式下，是否打乱序列顺序（不打乱帧内时序）
SHUFFLE_SEQUENCES = True

# 是否启用主干中的轻量时序注意力（默认关闭，避免替代ROI级融合）
ENABLE_TEMPORAL_ATTENTION = False

# 时序注意力参数
TEMPORAL_ATTENTION_HEADS = 8

# 是否启用 ROI 级时序融合（论文思路的核心迁移目标）
ENABLE_ROI_TEMPORAL_FUSION = True

# ROI 时序融合中参与跨帧注意力的最大 proposal 数量
ROI_TEMPORAL_TOPK = 256

# ROI 时序融合中的相似度掩码阈值
ROI_TEMPORAL_SIM_THRESH = 0.9

# 时序一致性辅助损失权重
TEMPORAL_LOSS_WEIGHT = 0.1

# 特征图尺寸 (基于 1024x1024 输入)
# stride 4:  256x256
# stride 8:  128x128
# stride 16: 64x64
# stride 32: 32x32
FEATURE_MAP_SIZES = {
    "0": 256,  # stride 4
    "1": 128,  # stride 8
    "2": 64,   # stride 16
    "3": 32,   # stride 32
}

# ======================== 训练配置 ========================
# 批次大小 (由于图像较大，建议较小的batch size)
BATCH_SIZE = 8

# 训练轮数
NUM_EPOCHS = 50

# 学习率配置
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 学习率调度器
LR_SCHEDULER_STEP_SIZE = 15
LR_SCHEDULER_GAMMA = 0.1

# 优化器选择: "adamw" 或 "sgd"
OPTIMIZER = "adamw"

# SGD 专用参数
SGD_MOMENTUM = 0.9

# 梯度裁剪
GRADIENT_CLIP_MAX_NORM = 10.0

# ======================== 数据增强配置 ========================
# ImageNet 归一化参数 (DINOv3 预训练使用)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 增强概率
AUGMENTATION_PROB = 0.5

# ======================== 其他配置 ========================
# DataLoader 配置
NUM_WORKERS = 4
PIN_MEMORY = True

# 日志打印频率 (每N个batch打印一次)
LOG_INTERVAL = 10

# 模型保存频率 (每N个epoch保存一次)
SAVE_INTERVAL = 5

# 设备配置
DEVICE = "cuda"  # "cuda" 或 "cpu"


def get_config_dict():
    """返回配置字典，方便日志记录"""
    return {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER,
        "num_classes": NUM_CLASSES,
        "adapter_out_channels": ADAPTER_OUT_CHANNELS,
        "dino_embed_dim": DINO_EMBED_DIM,
    }


if __name__ == "__main__":
    # 测试配置
    print("=" * 50)
    print("DINOv3 + Faster R-CNN 配置信息")
    print("=" * 50)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"DINOv3 仓库: {DINOV3_REPO_DIR}")
    print(f"DINOv3 权重: {DINOV3_WEIGHTS_PATH}")
    print(f"数据目录: {DATA_ROOT}")
    print(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Patch 数量: {DINO_NUM_PATCHES}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print("=" * 50)
