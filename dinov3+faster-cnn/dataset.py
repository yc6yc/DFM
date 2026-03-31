# -*- coding: utf-8 -*-
"""
数据集模块 - 血管狭窄检测
功能:
1. 从COCO格式文件加载数据集
2. 解析 Pascal VOC XML 标注
3. 在线数据增强 (Albumentations)
"""

import os
import json
import random
import xml.etree.ElementTree as ET
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def _parse_sequence_meta(file_name: str) -> Tuple[str, int]:
    """从 file_name 中解析序列ID与帧索引。"""
    path = Path(file_name)

    # 常见结构: <seq_id>/images/<seq_id>_<frame_idx>.jpg
    # 兼容 data03/<seq_id>/images/*.jpg 这种前缀目录形式
    if len(path.parts) >= 2 and path.parts[0].startswith("data"):
        sequence_id = path.parts[1]
    elif len(path.parts) > 0:
        sequence_id = path.parts[0]
    else:
        sequence_id = path.stem

    stem_parts = path.stem.split('_')
    frame_index = -1
    for token in reversed(stem_parts):
        if token.isdigit():
            frame_index = int(token)
            break

    return sequence_id, frame_index


class SequenceBatchSampler(Sampler[List[int]]):
    """按序列生成批次，支持每步聚合多个病例序列。"""

    def __init__(
        self,
        sequence_to_indices: Dict[str, List[int]],
        seq_len: int,
        seq_stride: int = 1,
        shuffle: bool = True,
        min_frames: int = 2,
        sequence_batch_size: int = 1,
        require_exact_seq_len: bool = False,
    ):
        self.sequence_to_indices = sequence_to_indices
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.shuffle = shuffle
        self.min_frames = max(1, min_frames)
        self.sequence_batch_size = max(1, sequence_batch_size)
        self.require_exact_seq_len = require_exact_seq_len
        self.sequence_units = self._build_sequence_units()

        if len(self.sequence_units) == 0:
            raise RuntimeError("未生成任何有效序列批次，请检查序列长度与数据集配置")

    def _build_sequence_units(self) -> List[List[int]]:
        sequence_units: List[List[int]] = []
        for seq_id, indices in self.sequence_to_indices.items():
            if len(indices) < self.min_frames:
                continue

            # 保证同病例内帧顺序严格有序
            sorted_indices = list(indices)

            if self.require_exact_seq_len and len(sorted_indices) != self.seq_len:
                raise RuntimeError(
                    f"病例 {seq_id} 帧数={len(sorted_indices)}，与要求的 seq_len={self.seq_len} 不一致"
                )

            if len(sorted_indices) < self.seq_len:
                # 兼容短序列：复制首帧前置补齐
                pad_count = self.seq_len - len(sorted_indices)
                sequence_units.append([sorted_indices[0]] * pad_count + sorted_indices)
                continue

            if len(sorted_indices) == self.seq_len:
                sequence_units.append(sorted_indices)
                continue

            for start in range(0, len(sorted_indices) - self.seq_len + 1, self.seq_stride):
                sequence_units.append(sorted_indices[start:start + self.seq_len])

            # 如果不能整除，补一个尾窗避免丢帧
            if (len(sorted_indices) - self.seq_len) % self.seq_stride != 0:
                sequence_units.append(sorted_indices[-self.seq_len:])

        return sequence_units

    def __iter__(self):
        units = list(self.sequence_units)
        if self.shuffle:
            random.shuffle(units)

        # 每个优化step聚合多个病例序列（例如4个序列）
        for start in range(0, len(units), self.sequence_batch_size):
            chunk = units[start:start + self.sequence_batch_size]
            batch = []
            for seq in chunk:
                batch.extend(seq)
            yield batch

    def __len__(self):
        return math.ceil(len(self.sequence_units) / self.sequence_batch_size)


# ======================== COCO数据加载 ========================
def load_coco_dataset(coco_json_path: Path) -> List[Dict]:
    """
    从COCO格式JSON文件加载数据集
    
    Args:
        coco_json_path: COCO格式JSON文件路径
        
    Returns:
        samples: 样本列表，每个元素包含图像路径和标注信息
    """
    print(f"加载COCO数据集: {coco_json_path}")
    
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 构建image_id到image信息的映射
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # 构建image_id到annotations的映射
    annotations_dict = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        annotations_dict[image_id].append(ann)
    
    # 构建样本列表
    samples = []
    for img_id, img_info in images_dict.items():
        # 获取图像路径（相对路径）
        file_name = img_info['file_name']
        sequence_id, frame_index = _parse_sequence_meta(file_name)
        
        # 获取标注
        anns = annotations_dict.get(img_id, [])
        
        # 转换COCO格式的bbox为VOC格式
        boxes = []
        labels = []
        for ann in anns:
            # COCO格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # 转换为VOC格式: [xmin, ymin, xmax, ymax]
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
        
        samples.append({
            'image_path': file_name,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'sequence_id': sequence_id,
            'frame_index': frame_index,
        })
    
    print(f"  加载完成: {len(samples)} 张图像")
    return samples


# ======================== 数据增强 ========================
def get_train_transforms():
    """
    训练集数据增强流水线
    适用于医疗影像的增强策略
    """
    return A.Compose([
        # 强制resize到目标尺寸
        A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
        
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 亮度/对比度增强 (对DSA图像很重要)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Gamma 校正 (调整图像整体亮度分布)
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        
        # 轻微的高斯噪声（移除mean参数以兼容新版本）
        A.GaussNoise(p=0.3),
        
        # 归一化 (使用 ImageNet 均值和标准差)
        A.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            max_pixel_value=255.0
        ),
        
        # 转换为 PyTorch Tensor
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # (xmin, ymin, xmax, ymax)
        label_fields=['labels'],
        min_visibility=0.3,  # 裁剪后至少保留30%的框
    ))


def get_val_transforms():
    """
    验证集数据变换 (仅归一化，不增强)
    注意：验证集不做几何变换，但仍需bbox_params以保持接口一致
    """
    return A.Compose([
        # 强制resize到目标尺寸
        A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
        
        A.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=0,
        min_visibility=0
    ))


# ======================== Dataset 类 ========================
class VascularStenosisDataset(Dataset):
    """
    血管狭窄检测数据集（COCO格式）
    
    数据格式:
    - 图像: 1024x1024 RGB JPG
    - 标注: COCO格式JSON
    """
    
    def __init__(
        self,
        data_root: Path,
        samples: List[Dict],
        transforms=None,
        is_train: bool = True
    ):
        """
        Args:
            data_root: 数据根目录 (data02/)
            samples: 样本列表（从COCO JSON加载）
            transforms: Albumentations 变换
            is_train: 是否为训练模式
        """
        self.data_root = data_root
        self.samples = samples
        self.transforms = transforms
        self.is_train = is_train

        # 建立序列到样本索引的映射（按帧序排序）
        self.sequence_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            self.sequence_to_indices[sample['sequence_id']].append(idx)

        for seq_id in self.sequence_to_indices:
            self.sequence_to_indices[seq_id].sort(
                key=lambda i: self.samples[i]['frame_index']
            )
        
        print(f"{'训练集' if is_train else '验证集'} 初始化完成: {len(self.samples)} 张图像")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回格式:
        - image: Tensor [3, 1024, 1024]
        - target: Dict {
            'boxes': Tensor [N, 4] (xmin, ymin, xmax, ymax)
            'labels': Tensor [N] (类别ID, 从1开始)
            'image_id': Tensor [1]
          }
        """
        sample = self.samples[idx]
        
        # 1. 读取图像
        # 优先使用项目根目录拼接 COCO 中的相对路径，避免误拼到错误数据目录
        img_path = config.PROJECT_ROOT / sample['image_path']
        if not img_path.exists():
            # 兼容标注里使用 data03 前缀、实际数据在 data02 的情况
            rel_path = Path(sample['image_path'])
            if len(rel_path.parts) >= 1 and rel_path.parts[0].startswith("data"):
                remapped = Path("data02").joinpath(*rel_path.parts[1:])
                remapped_abs = config.PROJECT_ROOT / remapped
                if remapped_abs.exists():
                    img_path = remapped_abs

        if not img_path.exists():
            # 最后兜底旧目录结构
            img_path = self.data_root.parent / sample['image_path']

        if not img_path.exists():
            raise FileNotFoundError(f"图像不存在: {sample['image_path']} (尝试路径: {img_path})")
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)  # [H, W, 3]
        
        # 2. 获取标注
        boxes = sample['boxes'].copy()
        labels = sample['labels'].copy()
        
        # 3. 确保坐标合法
        valid_boxes = []
        valid_labels = []
        
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            
            # 坐标修正
            xmin = max(0, min(xmin, config.IMAGE_SIZE - 1))
            ymin = max(0, min(ymin, config.IMAGE_SIZE - 1))
            xmax = max(0, min(xmax, config.IMAGE_SIZE))
            ymax = max(0, min(ymax, config.IMAGE_SIZE))
            
            # 跳过无效框
            if xmax <= xmin or ymax <= ymin:
                continue
            
            valid_boxes.append([xmin, ymin, xmax, ymax])
            valid_labels.append(label)
        
        # 4. 应用数据增强
        if self.transforms is not None and len(valid_boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=valid_boxes,
                labels=valid_labels
            )
            image = transformed['image']
            valid_boxes = transformed['bboxes']
            valid_labels = transformed['labels']
        else:
            # 如果没有增强，仍需要归一化和转换为tensor
            transform = A.Compose([
                A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
                ToTensorV2()
            ])
            image = transform(image=image)['image']
        
        # 5. 转换为 Tensor
        if len(valid_boxes) == 0:
            # 如果没有目标框，返回空
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(valid_labels, dtype=torch.int64)
        
        # 6. 构建 target 字典
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([sample['image_id']]),
            'sequence_id': sample['sequence_id'],
            'frame_index': torch.tensor([sample['frame_index']], dtype=torch.int64),
        }
        
        return image, target


# ======================== Collate 函数 ========================
def collate_fn(batch):
    """
    自定义 collate 函数，处理变长的 bounding boxes
    
    Args:
        batch: List of (image, target)
        
    Returns:
        images: Tuple of Tensors
        targets: Tuple of Dicts
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # 不需要堆叠，Faster R-CNN 接受 tuple of tensors
    return tuple(images), tuple(targets)


# ======================== DataLoader 构建 ========================
def build_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练集和验证集的 DataLoader
    
    Returns:
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
    """
    print("=" * 50)
    print("构建数据加载器")
    print("=" * 50)
    
    # 1. 从COCO JSON文件加载数据集
    train_samples = load_coco_dataset(config.TRAIN_COCO_JSON)
    val_samples = load_coco_dataset(config.VAL_COCO_JSON)
    
    # 2. 创建 Dataset
    train_dataset = VascularStenosisDataset(
        data_root=config.DATA_ROOT,
        samples=train_samples,
        transforms=get_train_transforms(),
        is_train=True
    )
    
    val_dataset = VascularStenosisDataset(
        data_root=config.DATA_ROOT,
        samples=val_samples,
        transforms=get_val_transforms(),
        is_train=False
    )
    
    # 3. 创建 DataLoader
    if config.ENABLE_TEMPORAL_TRAINING:
        print("启用时序训练模式: 使用按序列批采样")
        print(f"  序列长度: {config.SEQUENCE_LENGTH}")
        print(f"  每step序列数(时序batch): {batch_size}")

        train_batch_sampler = SequenceBatchSampler(
            sequence_to_indices=train_dataset.sequence_to_indices,
            seq_len=config.SEQUENCE_LENGTH,
            seq_stride=config.SEQUENCE_STRIDE,
            shuffle=config.SHUFFLE_SEQUENCES,
            min_frames=config.TEMPORAL_MIN_FRAMES,
            sequence_batch_size=batch_size,
            require_exact_seq_len=config.TEMPORAL_REQUIRE_EXACT_FRAMES,
        )

        val_batch_sampler = SequenceBatchSampler(
            sequence_to_indices=val_dataset.sequence_to_indices,
            seq_len=config.SEQUENCE_LENGTH,
            seq_stride=config.SEQUENCE_STRIDE,
            shuffle=False,
            min_frames=config.TEMPORAL_MIN_FRAMES,
            sequence_batch_size=1,
            require_exact_seq_len=config.TEMPORAL_REQUIRE_EXACT_FRAMES,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=config.PIN_MEMORY,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=config.PIN_MEMORY,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=config.PIN_MEMORY
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=config.PIN_MEMORY
        )
    
    print("=" * 50)
    print(f"数据加载器构建完成")
    print(f"  训练集: {len(train_dataset)} 张图像")
    print(f"  验证集: {len(val_dataset)} 张图像")
    print("=" * 50)
    
    return train_loader, val_loader


# ======================== 测试代码 ========================
if __name__ == "__main__":
    print("=" * 50)
    print("测试数据集加载")
    print("=" * 50)
    
    # 构建 DataLoader
    train_loader, val_loader = build_dataloaders(batch_size=2, num_workers=0)
    
    # 测试一个 batch
    images, targets = next(iter(train_loader))
    
    print(f"\nBatch 信息:")
    print(f"  图像数量: {len(images)}")
    print(f"  图像形状: {images[0].shape}")
    print(f"  图像数据类型: {images[0].dtype}")
    print(f"  图像范围: [{images[0].min():.3f}, {images[0].max():.3f}]")
    
    print(f"\n标注信息:")
    for i, target in enumerate(targets):
        print(f"  图像 {i}:")
        print(f"    boxes: {target['boxes'].shape} - {target['boxes']}")
        print(f"    labels: {target['labels'].shape} - {target['labels']}")
    
    print("\n数据集测试完成！")
