# -*- coding: utf-8 -*-
"""
训练模块 - DINOv3 + Faster R-CNN 血管狭窄检测
功能:
1. 训练循环
2. 验证循环
3. 模型保存/加载
4. 日志记录
"""

import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import config
from dataset import build_dataloaders
from model import build_faster_rcnn_model


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无效布尔值: {value}")


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


# ======================== 训练器类 ========================
class Trainer:
    """
    训练器封装所有训练逻辑
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        checkpoint_dir,
        log_dir,
        max_train_batches=None,
        max_val_batches=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # 日志记录
        self.train_losses = []
        self.val_losses = []
        
        # 创建日志文件
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # TensorBoard
        tensorboard_dir = self.log_dir / 'tensorboard'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        self.log(f"TensorBoard 日志目录: {tensorboard_dir}")

    def _evaluate_ap(self, preds, gt_by_image, class_ids: List[int], iou_thresh: float) -> float:
        ap_list = []

        for cls in class_ids:
            gt_cls = defaultdict(list)
            for img_id, anns in gt_by_image.items():
                boxes = [a["bbox"] for a in anns if int(a["category_id"]) == cls]
                gt_cls[img_id] = boxes

            num_gt = sum(len(v) for v in gt_cls.values())
            if num_gt == 0:
                continue

            dets = [p for p in preds if int(p["category_id"]) == cls]
            dets.sort(key=lambda x: x["score"], reverse=True)

            gt_used = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_cls.items()}
            tp = np.zeros(len(dets), dtype=np.float32)
            fp = np.zeros(len(dets), dtype=np.float32)

            for i, d in enumerate(dets):
                img_id = int(d["image_id"])
                pred_box = d["bbox"]
                gts = gt_cls.get(img_id, [])

                if len(gts) == 0:
                    fp[i] = 1.0
                    continue

                ious = [box_iou_xyxy(pred_box, g) for g in gts]
                best_idx = int(np.argmax(ious)) if len(ious) > 0 else -1
                best_iou = ious[best_idx] if best_idx >= 0 else 0.0

                if best_iou >= iou_thresh and not gt_used[img_id][best_idx]:
                    tp[i] = 1.0
                    gt_used[img_id][best_idx] = True
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / max(1, num_gt)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap_list.append(compute_ap(recalls, precisions))

        if len(ap_list) == 0:
            return 0.0
        return float(np.mean(ap_list))

    def _evaluate_prf1_at_iou50(self, preds, gt_by_image, class_ids: List[int]):
        valid_preds = [p for p in preds if int(p["category_id"]) in class_ids]
        valid_preds.sort(key=lambda x: x["score"], reverse=True)

        gt_cls = defaultdict(list)
        for img_id, anns in gt_by_image.items():
            for a in anns:
                cid = int(a["category_id"])
                if cid in class_ids:
                    gt_cls[img_id].append({"category_id": cid, "bbox": a["bbox"]})

        total_gt = sum(len(v) for v in gt_cls.values())
        if total_gt == 0:
            return 0.0, 0.0, 0.0, 0.0

        gt_used = {img_id: np.zeros(len(anns), dtype=bool) for img_id, anns in gt_cls.items()}
        tp = np.zeros(len(valid_preds), dtype=np.float32)
        fp = np.zeros(len(valid_preds), dtype=np.float32)

        for i, d in enumerate(valid_preds):
            img_id = int(d["image_id"])
            cid = int(d["category_id"])
            pbox = d["bbox"]
            gts = gt_cls.get(img_id, [])

            if len(gts) == 0:
                fp[i] = 1.0
                continue

            candidate_indices = [j for j, g in enumerate(gts) if int(g["category_id"]) == cid]
            if len(candidate_indices) == 0:
                fp[i] = 1.0
                continue

            best_iou = 0.0
            best_j = -1
            for j in candidate_indices:
                iou = box_iou_xyxy(pbox, gts[j]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.5 and best_j >= 0 and not gt_used[img_id][best_j]:
                tp[i] = 1.0
                gt_used[img_id][best_j] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        recall_curve = tp_cum / max(1, total_gt)
        f1_curve = 2.0 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-12)

        if len(f1_curve) == 0:
            return 0.0, 0.0, 0.0, 0.0

        best_idx = int(np.argmax(f1_curve))
        return (
            float(precision_curve[best_idx]),
            float(recall_curve[best_idx]),
            float(f1_curve[best_idx]),
            float(valid_preds[best_idx]["score"]),
        )

    @torch.no_grad()
    def _evaluate_detection_metrics(self):
        self.model.eval()

        preds = []
        gt_by_image = defaultdict(list)
        class_ids = list(range(1, config.NUM_CLASSES))
        num_batches = len(self.val_loader)

        for batch_idx, (images, targets) in enumerate(self.val_loader):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break

            images = [img.to(self.device) for img in images]
            outputs = self.model(images)

            for t, out in zip(targets, outputs):
                image_id = int(t["image_id"].reshape(-1)[0].item())

                gt_boxes = t["boxes"].detach().cpu().numpy().tolist()
                gt_labels = t["labels"].detach().cpu().numpy().tolist()
                for b, c in zip(gt_boxes, gt_labels):
                    gt_by_image[image_id].append({"category_id": int(c), "bbox": [float(v) for v in b]})

                boxes = out["boxes"].detach().cpu().numpy().tolist()
                scores = out["scores"].detach().cpu().numpy().tolist()
                labels = out["labels"].detach().cpu().numpy().tolist()
                for b, s, c in zip(boxes, scores, labels):
                    preds.append(
                        {
                            "image_id": image_id,
                            "category_id": int(c),
                            "bbox": [float(v) for v in b],
                            "score": float(s),
                        }
                    )

        precision, recall, f1, best_conf = self._evaluate_prf1_at_iou50(preds, gt_by_image, class_ids)
        map_50 = self._evaluate_ap(preds, gt_by_image, class_ids, iou_thresh=0.5)
        map_75 = self._evaluate_ap(preds, gt_by_image, class_ids, iou_thresh=0.75)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_confidence_for_f1": best_conf,
            "mAP@0.5": map_50,
            "mAP@0.75": map_75,
        }
    
    def log(self, message):
        """记录日志到文件和控制台"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def train_one_epoch(self, epoch):
        """
        训练一个 epoch
        
        Args:
            epoch: 当前 epoch 编号
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        epoch_loss = 0.0
        loss_components = {}
        
        num_batches = len(self.train_loader)
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break

            # 将数据移到设备上
            images = [img.to(self.device) for img in images]
            targets = [
                {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                for t in targets
            ]
            
            # 前向传播
            loss_dict = self.model(images, targets)
            
            # 计算总损失
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            self.optimizer.zero_grad()
            losses.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.GRADIENT_CLIP_MAX_NORM
            )
            
            # 优化器步进
            self.optimizer.step()
            
            # 记录损失
            epoch_loss += losses.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            
            # 定期打印日志
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                batch_time = elapsed / (batch_idx + 1)
                eta = batch_time * (num_batches - batch_idx - 1)
                
                self.log(
                    f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                    f"Batch [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {losses.item():.4f} "
                    f"ETA: {eta:.0f}s"
                )
        
        # 计算平均损失
        effective_batches = min(num_batches, self.max_train_batches) if self.max_train_batches is not None else num_batches
        effective_batches = max(1, effective_batches)
        avg_loss = epoch_loss / effective_batches
        for key in loss_components.keys():
            loss_components[key] /= effective_batches
        
        # 记录详细损失
        self.log(
            f"\nEpoch [{epoch}/{config.NUM_EPOCHS}] 训练完成:"
        )
        self.log(f"  总损失: {avg_loss:.4f}")
        for key, value in loss_components.items():
            self.log(f"  {key}: {value:.4f}")
        
        # TensorBoard 记录训练损失
        self.writer.add_scalar('Loss/train_total', avg_loss, epoch)
        for key, value in loss_components.items():
            self.writer.add_scalar(f'Loss/train_{key}', value, epoch)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """
        验证模型
        
        Args:
            epoch: 当前 epoch 编号
            
        Returns:
            avg_loss: 平均验证损失
        """
        self.model.train()  # Faster R-CNN 在验证时也需要 train 模式才能计算 loss
        epoch_loss = 0.0
        num_batches = len(self.val_loader)
        
        for batch_idx, (images, targets) in enumerate(self.val_loader):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break

            # 将数据移到设备上
            images = [img.to(self.device) for img in images]
            targets = [
                {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                for t in targets
            ]
            
            # 前向传播
            with torch.set_grad_enabled(True):  # 需要梯度才能计算 loss
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            epoch_loss += losses.item()
        
        effective_batches = min(num_batches, self.max_val_batches) if self.max_val_batches is not None else num_batches
        effective_batches = max(1, effective_batches)
        avg_loss = epoch_loss / effective_batches
        
        self.log(
            f"Epoch [{epoch}/{config.NUM_EPOCHS}] 验证完成:"
        )
        self.log(f"  验证损失: {avg_loss:.4f}")

        metrics = self._evaluate_detection_metrics()
        self.log(
            "  检测指标: "
            f"Prec={metrics['precision']:.6f}, "
            f"Rec={metrics['recall']:.6f}, "
            f"F1={metrics['f1']:.6f}, "
            f"mAP@0.5={metrics['mAP@0.5']:.6f}, "
            f"mAP@0.75={metrics['mAP@0.75']:.6f}"
        )
        
        # TensorBoard 记录验证损失
        self.writer.add_scalar('Loss/val_total', avg_loss, epoch)
        self.writer.add_scalar('Metrics/precision', metrics['precision'], epoch)
        self.writer.add_scalar('Metrics/recall', metrics['recall'], epoch)
        self.writer.add_scalar('Metrics/f1', metrics['f1'], epoch)
        self.writer.add_scalar('Metrics/mAP@0.5', metrics['mAP@0.5'], epoch)
        self.writer.add_scalar('Metrics/mAP@0.75', metrics['mAP@0.75'], epoch)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存模型检查点
        
        Args:
            epoch: 当前 epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # 保存最新模型
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        self.log(f"模型已保存: {latest_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.log(f"最佳模型已保存: {best_path}")
        
        # 定期保存里程碑模型
        if epoch % config.SAVE_INTERVAL == 0:
            milestone_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
            torch.save(checkpoint, milestone_path)
            self.log(f"里程碑模型已保存: {milestone_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        if not Path(checkpoint_path).exists():
            self.log(f"警告: 检查点不存在: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        self.log(f"检查点已加载: {checkpoint_path}")
        self.log(f"从 Epoch {self.current_epoch} 继续训练")
    
    def train(self, num_epochs):
        """
        完整的训练流程
        
        Args:
            num_epochs: 训练轮数
        """
        self.log("=" * 50)
        self.log("开始训练")
        self.log("=" * 50)
        self.log(f"训练集大小: {len(self.train_loader.dataset)}")
        self.log(f"验证集大小: {len(self.val_loader.dataset)}")
        self.log(f"批次大小: {config.BATCH_SIZE}")
        self.log(f"训练轮数: {num_epochs}")
        self.log(f"学习率: {config.LEARNING_RATE}")
        self.log(f"设备: {self.device}")
        self.log("=" * 50)
        
        start_epoch = self.current_epoch + 1
        
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_one_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.log(f"学习率: {current_lr:.6f}")
            
            # TensorBoard 记录学习率
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 保存模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.log(f"✓ 新的最佳验证损失: {self.best_val_loss:.4f}")
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # 记录 epoch 耗时
            epoch_time = time.time() - epoch_start_time
            self.log(f"Epoch {epoch} 耗时: {epoch_time:.0f}s")
            self.log("-" * 50)
        
        self.log("=" * 50)
        self.log("训练完成！")
        self.log(f"最佳验证损失: {self.best_val_loss:.4f}")
        self.log("=" * 50)
        
        # 保存训练曲线数据
        self.save_training_history()
        
        # 关闭 TensorBoard writer
        self.writer.close()
        self.log("TensorBoard 日志已保存")
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        self.log(f"训练历史已保存: {history_path}")


# ======================== 主训练函数 ========================
def main():
    """
    主训练流程
    """
    parser = argparse.ArgumentParser(description="DINOv3 + Faster R-CNN 训练入口")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="训练轮数覆盖配置")
    parser.add_argument("--max-train-batches", type=int, default=None, help="每个epoch最多训练batch数")
    parser.add_argument("--max-val-batches", type=int, default=None, help="每个epoch最多验证batch数")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader worker数量")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="非时序模式下batch大小")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--lr-scheduler", type=str, choices=["step", "cosine"], default="cosine", help="学习率调度策略")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="余弦调度最小学习率")
    parser.add_argument("--enable-temporal-training", type=_str2bool, default=None, help="是否启用时序采样")
    parser.add_argument("--enable-roi-temporal-fusion", type=_str2bool, default=None, help="是否启用ROI时序融合")
    parser.add_argument("--temporal-loss-weight", type=float, default=None, help="时序一致性损失权重")
    parser.add_argument("--sequence-length", type=int, default=None, help="时序采样窗口长度")
    parser.add_argument("--sequence-stride", type=int, default=None, help="时序滑窗步长")
    parser.add_argument("--temporal-min-frames", type=int, default=None, help="时序最小帧数")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="覆盖checkpoint输出目录")
    parser.add_argument("--log-dir", type=str, default=None, help="覆盖日志输出目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复完整训练状态的checkpoint路径")
    parser.add_argument("--init-weights", type=str, default=None, help="仅加载模型权重的checkpoint路径（非严格）")
    args = parser.parse_args()

    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.enable_temporal_training is not None:
        config.ENABLE_TEMPORAL_TRAINING = args.enable_temporal_training
    if args.enable_roi_temporal_fusion is not None:
        config.ENABLE_ROI_TEMPORAL_FUSION = args.enable_roi_temporal_fusion
    if args.temporal_loss_weight is not None:
        config.TEMPORAL_LOSS_WEIGHT = args.temporal_loss_weight
    if args.sequence_length is not None:
        config.SEQUENCE_LENGTH = args.sequence_length
    if args.sequence_stride is not None:
        config.SEQUENCE_STRIDE = args.sequence_stride
    if args.temporal_min_frames is not None:
        config.TEMPORAL_MIN_FRAMES = args.temporal_min_frames

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.CHECKPOINT_DIR
    log_dir = Path(args.log_dir) if args.log_dir else config.LOG_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("DINOv3 + Faster R-CNN 血管狭窄检测")
    print("=" * 50)
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 构建数据加载器
    print("\n构建数据加载器...")
    train_loader, val_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 构建模型
    print("\n构建模型...")
    model = build_faster_rcnn_model(num_classes=config.NUM_CLASSES)
    model = model.to(device)

    if args.init_weights:
        print(f"\n加载初始化权重: {args.init_weights}")
        init_ckpt = torch.load(args.init_weights, map_location=device)
        state_dict = init_ckpt.get('model_state_dict', init_ckpt)
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"  missing keys: {len(load_result.missing_keys)}")
        print(f"  unexpected keys: {len(load_result.unexpected_keys)}")
    
    # 构建优化器
    print("\n配置优化器...")
    # 只优化可训练参数
    params = [p for p in model.parameters() if p.requires_grad]
    
    if config.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        print(f"优化器: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    else:
        optimizer = optim.SGD(
            params,
            lr=config.LEARNING_RATE,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        print(f"优化器: SGD (lr={config.LEARNING_RATE}, momentum={config.SGD_MOMENTUM})")
    
    # 学习率调度器
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
        print(f"学习率调度器: CosineAnnealingLR (T_max={max(1, args.epochs)}, eta_min={args.min_lr})")
    else:
        scheduler = StepLR(
            optimizer,
            step_size=config.LR_SCHEDULER_STEP_SIZE,
            gamma=config.LR_SCHEDULER_GAMMA
        )
        print(f"学习率调度器: StepLR (step={config.LR_SCHEDULER_STEP_SIZE}, gamma={config.LR_SCHEDULER_GAMMA})")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(num_epochs=args.epochs)
    
    print("\n训练完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise
