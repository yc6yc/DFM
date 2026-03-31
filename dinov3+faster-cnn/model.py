# -*- coding: utf-8 -*-
"""
模型模块 - DINOv3 + Faster R-CNN
功能:
1. 加载 DINOv3 预训练模型
2. 构建 Adapter 适配器（特征金字塔）
3. 集成 Faster R-CNN 检测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import re
from pathlib import Path
from collections import OrderedDict
from collections import defaultdict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import config


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_prefix_from_name(path: Path) -> str | None:
    match = re.search(r"-([0-9a-fA-F]{8,})\.pth$", path.name)
    if match is None:
        return None
    return match.group(1).lower()


def _ensure_hub_cache_consistency(weights_path: Path) -> None:
    """清理与本地权重不一致的 torch.hub 缓存，避免命中损坏旧文件。"""
    weights_path = weights_path.expanduser().resolve()
    if not weights_path.exists():
        return

    hub_ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cached_path = hub_ckpt_dir / weights_path.name
    if not cached_path.exists():
        return

    # 若缓存路径与源路径是同一文件，则无需处理。
    try:
        if cached_path.resolve() == weights_path:
            return
    except OSError:
        pass

    src_size = weights_path.stat().st_size
    cache_size = cached_path.stat().st_size

    if src_size != cache_size:
        print(f"检测到不一致的 torch.hub 缓存，已删除: {cached_path}")
        cached_path.unlink()
        return

    hash_prefix = _hash_prefix_from_name(weights_path)
    if hash_prefix is None:
        return

    src_hash = _file_sha256(weights_path)
    if not src_hash.startswith(hash_prefix):
        raise RuntimeError(
            f"本地权重文件哈希校验失败: {weights_path} (期望前缀 {hash_prefix}, 实际 {src_hash[:8]})"
        )

    cache_hash = _file_sha256(cached_path)
    if cache_hash != src_hash:
        print(f"检测到损坏的 torch.hub 缓存，已删除: {cached_path}")
        cached_path.unlink()


class TemporalROIFusion(nn.Module):
    """对 Faster R-CNN 的 ROI 特征执行跨帧时序注意力融合。"""

    def __init__(self, feat_dim: int, num_heads: int = 8, topk: int = 256, sim_thresh: float = 0.9):
        super().__init__()
        self.feat_dim = feat_dim
        heads = max(1, min(num_heads, feat_dim))
        while heads > 1 and feat_dim % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.topk = topk
        self.sim_thresh = sim_thresh

        self.proposal_score = nn.Linear(feat_dim, 1)
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim * 2, feat_dim)
        self.dropout = nn.Dropout(p=0.1)

        # 由 RoIHeads.forward 注入的上下文
        self._proposal_counts = None
        self._sequence_ids = None
        self._frame_indices = None
        self._last_temporal_loss = None

    def set_context(self, proposal_counts, targets=None):
        self._proposal_counts = [int(c) for c in proposal_counts]

        if targets is None:
            self._sequence_ids = ["__default_seq__"] * len(self._proposal_counts)
            self._frame_indices = list(range(len(self._proposal_counts)))
            return

        sequence_ids = []
        frame_indices = []
        for i, target in enumerate(targets):
            seq_id = target.get("sequence_id", "__default_seq__")
            if isinstance(seq_id, torch.Tensor):
                seq_id = str(int(seq_id.reshape(-1)[0].item()))
            else:
                seq_id = str(seq_id)

            frame_idx = target.get("frame_index", i)
            if isinstance(frame_idx, torch.Tensor):
                frame_idx = int(frame_idx.reshape(-1)[0].item())
            else:
                try:
                    frame_idx = int(frame_idx)
                except (TypeError, ValueError):
                    frame_idx = i

            sequence_ids.append(seq_id)
            frame_indices.append(frame_idx)

        self._sequence_ids = sequence_ids
        self._frame_indices = frame_indices

    def clear_context(self):
        self._proposal_counts = None
        self._sequence_ids = None
        self._frame_indices = None

    def pop_last_temporal_loss(self):
        loss = self._last_temporal_loss
        self._last_temporal_loss = None
        return loss

    def _cross_frame_attention(self, src_feats: torch.Tensor, dst_feats: torch.Tensor) -> torch.Tensor:
        if src_feats.shape[0] == 0 or dst_feats.shape[0] == 0:
            return dst_feats

        head_dim = self.feat_dim // self.num_heads

        q = self.q_proj(dst_feats).reshape(-1, self.num_heads, head_dim).transpose(0, 1)
        k = self.k_proj(src_feats).reshape(-1, self.num_heads, head_dim).transpose(0, 1)
        v = self.v_proj(src_feats).reshape(-1, self.num_heads, head_dim).transpose(0, 1)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.matmul(qn, kn.transpose(-2, -1))
        valid_mask = sim > self.sim_thresh

        logits = logits.masked_fill(~valid_mask, -1e4)
        attn = logits.softmax(dim=-1)

        # 若某个目标 proposal 没有任何可匹配项，则令其注意力输出为0
        has_match = valid_mask.any(dim=-1, keepdim=True)
        attn = torch.where(has_match, attn, torch.zeros_like(attn))
        attn = self.dropout(attn)

        aggregated = torch.matmul(attn, v).transpose(0, 1).reshape(dst_feats.shape[0], self.feat_dim)
        fused = self.out_proj(torch.cat([aggregated, dst_feats], dim=-1))
        return fused

    def _global_fusion_fallback(self, roi_features: torch.Tensor) -> torch.Tensor:
        num_tokens = roi_features.shape[0]
        keep_k = min(self.topk, num_tokens)
        if keep_k < 2:
            return roi_features

        scores = self.proposal_score(roi_features).squeeze(-1)
        topk_indices = torch.topk(scores, k=keep_k, dim=0).indices
        selected = roi_features[topk_indices]
        fused = self._cross_frame_attention(selected, selected)
        self._last_temporal_loss = F.mse_loss(fused, selected)

        output = roi_features.clone()
        output[topk_indices] = 0.5 * selected + 0.5 * fused
        return output

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        self._last_temporal_loss = None

        if roi_features is None or roi_features.ndim != 2 or roi_features.shape[0] < 2:
            return roi_features

        # 没有上下文时退化为全局融合，保证推理兼容
        if self._proposal_counts is None:
            return self._global_fusion_fallback(roi_features)

        if sum(self._proposal_counts) != roi_features.shape[0]:
            return self._global_fusion_fallback(roi_features)

        if self._sequence_ids is None or self._frame_indices is None:
            return self._global_fusion_fallback(roi_features)

        output = roi_features.clone()
        temporal_losses = []

        offsets = [0]
        for c in self._proposal_counts:
            offsets.append(offsets[-1] + c)

        image_meta = []
        for i, count in enumerate(self._proposal_counts):
            if count <= 0:
                continue
            image_meta.append({
                "img_idx": i,
                "seq": self._sequence_ids[i] if i < len(self._sequence_ids) else "__default_seq__",
                "frame": self._frame_indices[i] if i < len(self._frame_indices) else i,
                "start": offsets[i],
                "end": offsets[i + 1],
            })

        if len(image_meta) < 2:
            return output

        by_sequence = defaultdict(list)
        for item in image_meta:
            by_sequence[item["seq"]].append(item)

        for _, frames in by_sequence.items():
            if len(frames) < 2:
                continue

            frames = sorted(frames, key=lambda x: x["frame"])

            for j in range(1, len(frames)):
                prev_f = frames[j - 1]
                curr_f = frames[j]

                prev_feats = output[prev_f["start"]:prev_f["end"]]
                curr_feats = output[curr_f["start"]:curr_f["end"]]

                if prev_feats.shape[0] == 0 or curr_feats.shape[0] == 0:
                    continue

                keep_k = min(self.topk, prev_feats.shape[0], curr_feats.shape[0])
                if keep_k < 2:
                    continue

                prev_scores = self.proposal_score(prev_feats).squeeze(-1)
                curr_scores = self.proposal_score(curr_feats).squeeze(-1)

                prev_idx = torch.topk(prev_scores, k=keep_k, dim=0).indices
                curr_idx = torch.topk(curr_scores, k=keep_k, dim=0).indices

                prev_sel = prev_feats[prev_idx]
                curr_sel = curr_feats[curr_idx]

                # 双向融合：prev->curr 与 curr->prev
                curr_fused = self._cross_frame_attention(prev_sel, curr_sel)
                prev_fused = self._cross_frame_attention(curr_sel, prev_sel)

                # proposal级时序一致性: 相邻帧 top-k proposal 融合后应保持接近
                temporal_losses.append(F.mse_loss(curr_fused, prev_sel))
                temporal_losses.append(F.mse_loss(prev_fused, curr_sel))

                blended_curr = 0.5 * curr_sel + 0.5 * curr_fused
                blended_prev = 0.5 * prev_sel + 0.5 * prev_fused

                global_curr_idx = curr_idx + curr_f["start"]
                global_prev_idx = prev_idx + prev_f["start"]

                output = output.index_copy(0, global_curr_idx, blended_curr)
                output = output.index_copy(0, global_prev_idx, blended_prev)

        # 保底路径：若严格序列匹配未产生损失，退化为相邻图像top-1 proposal一致性约束
        if not temporal_losses and len(image_meta) >= 2:
            ordered_meta = sorted(image_meta, key=lambda x: x["img_idx"])
            for j in range(1, len(ordered_meta)):
                a = ordered_meta[j - 1]
                b = ordered_meta[j]

                a_feats = output[a["start"]:a["end"]]
                b_feats = output[b["start"]:b["end"]]
                if a_feats.shape[0] == 0 or b_feats.shape[0] == 0:
                    continue

                a_idx = torch.argmax(self.proposal_score(a_feats).squeeze(-1)).reshape(1)
                b_idx = torch.argmax(self.proposal_score(b_feats).squeeze(-1)).reshape(1)

                temporal_losses.append(F.mse_loss(a_feats[a_idx], b_feats[b_idx]))

        if temporal_losses:
            self._last_temporal_loss = torch.stack(temporal_losses).mean()

        return output


class TemporalBoxHead(nn.Module):
    """包装原始 box_head，在 ROI 特征层插入时序融合。"""

    def __init__(self, base_box_head: nn.Module, temporal_fusion: TemporalROIFusion):
        super().__init__()
        self.base_box_head = base_box_head
        self.temporal_fusion = temporal_fusion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        roi_features = self.base_box_head(x)
        roi_features = self.temporal_fusion(roi_features)
        return roi_features


class TemporalChannelAttention(nn.Module):
    """对同一序列帧在通道维进行轻量时序注意力。"""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=max(1, min(num_heads, channels)),
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # feat: [T, C, H, W]，T 为同序列帧数
        if feat.shape[0] < 2:
            pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
            return feat, pooled

        pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)  # [T, C]
        tokens = pooled.unsqueeze(0)  # [1, T, C]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.squeeze(0)  # [T, C]
        gate = self.gate(attn_out).unsqueeze(-1).unsqueeze(-1)  # [T, C, 1, 1]
        feat_refined = feat * (1.0 + gate)
        return feat_refined, attn_out


# ======================== DINOv3 Backbone + Adapter ========================
class DinoV3BackboneWithAdapter(nn.Module):
    """
    DINOv3 主干 + 适配器
    
    流程:
    1. DINOv3 提取特征 -> [B, 4096, 768]
    2. Reshape 为特征图 -> [B, 768, 64, 64]
    3. Adapter 生成多尺度特征金字塔:
       - "0": [B, 256, 256, 256] (stride 4)
       - "1": [B, 256, 128, 128] (stride 8)
       - "2": [B, 256, 64, 64]   (stride 16)
       - "3": [B, 256, 32, 32]   (stride 32)
    """
    
    def __init__(
        self,
        dino_model,
        embed_dim: int = config.DINO_EMBED_DIM,
        out_channels: int = config.ADAPTER_OUT_CHANNELS,
        freeze_backbone: bool = True,
        layer_indices=None,
    ):
        """
        Args:
            dino_model: 预训练的 DINOv3 模型
            embed_dim: DINOv3 嵌入维度 (768 for ViT-B)
            out_channels: 输出特征通道数 (256 for Faster R-CNN)
            freeze_backbone: 是否冻结主干网络
        """
        super().__init__()
        
        self.dino_backbone = dino_model
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.fpn_levels = ["0", "1", "2", "3"]
        self.layer_indices = self._resolve_layer_indices(layer_indices)
        self.enable_temporal_attention = config.ENABLE_TEMPORAL_ATTENTION
        self._last_temporal_tokens = []

        print(f"使用 DINO 中间层索引: {self.layer_indices}")
        
        # 冻结 DINOv3 参数
        if freeze_backbone:
            print("冻结 DINOv3 主干网络参数...")
            for param in self.dino_backbone.parameters():
                param.requires_grad = False
            self.dino_backbone.eval()  # 设置为评估模式
        
        # ============ Adapter 层定义 ============
        # 每个DINO层各自做1x1通道对齐，避免单层重复上采样造成信息瓶颈
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in self.fpn_levels
        ])

        # 融合后的平滑卷积
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in self.fpn_levels
        ])

        if self.enable_temporal_attention:
            self.temporal_attn = nn.ModuleList([
                TemporalChannelAttention(
                    channels=out_channels,
                    num_heads=config.TEMPORAL_ATTENTION_HEADS,
                )
                for _ in self.fpn_levels
            ])
        
        # 记录输出特征图通道数 (Faster R-CNN 需要)
        self.out_channels_list = [out_channels] * 4

    def _resolve_layer_indices(self, layer_indices):
        """解析并验证用于FPN融合的DINO层索引。"""
        if layer_indices is None:
            layer_indices = config.DINO_FPN_LAYER_INDICES

        n_levels = len(config.FEATURE_MAP_SIZES)
        n_blocks = getattr(self.dino_backbone, "n_blocks", None)

        if layer_indices is None:
            if n_blocks is None:
                raise ValueError("无法自动推断DINO层索引：主干缺少 n_blocks 属性")
            if n_levels == 1:
                return [n_blocks - 1]
            # 均匀采样到各FPN层，覆盖浅层到深层
            return [round(i * (n_blocks - 1) / (n_levels - 1)) for i in range(n_levels)]

        if len(layer_indices) != n_levels:
            raise ValueError(
                f"DINO_FPN_LAYER_INDICES 长度必须为 {n_levels}，当前为 {len(layer_indices)}"
            )

        if n_blocks is not None:
            for idx in layer_indices:
                if idx < 0 or idx >= n_blocks:
                    raise ValueError(f"层索引 {idx} 超出范围 [0, {n_blocks - 1}]")

        return list(layer_indices)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, 1024, 1024]
            
        Returns:
            features: OrderedDict {
                "0": [B, 256, 256, 256]  # stride 4
                "1": [B, 256, 128, 128]  # stride 8
                "2": [B, 256, 64, 64]    # stride 16
                "3": [B, 256, 32, 32]    # stride 32
            }
        """
        B = x.shape[0]
        
        # ============ 步骤1: DINOv3 多层特征提取 ============
        with torch.no_grad():
            # 通过层索引一次提取多层 patch token
            outputs = self.dino_backbone.get_intermediate_layers(
                x, 
                n=self.layer_indices,
                reshape=False,
                return_class_token=False,
                norm=True
            )

        if len(outputs) != len(self.fpn_levels):
            raise RuntimeError(
                f"DINO 返回层数与 FPN 层数不一致: {len(outputs)} vs {len(self.fpn_levels)}"
            )

        # ============ 步骤2: 将每层 token reshape 为 2D 特征图 ============
        patch_hw = int(config.DINO_NUM_PATCHES ** 0.5)
        dino_feats_2d = []
        for layer_tokens in outputs:
            expected_shape = (B, config.DINO_NUM_PATCHES, self.embed_dim)
            if layer_tokens.shape != expected_shape:
                raise RuntimeError(
                    f"DINO 层输出形状错误: 期望 {expected_shape}，实际 {layer_tokens.shape}"
                )
            feat_2d = layer_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, patch_hw, patch_hw)
            dino_feats_2d.append(feat_2d)

        # ============ 步骤3: 按FPN尺度对齐并做横向分支 ============
        # 约定: layer_indices 从浅到深，对应 FPN "0" 到 "3"
        lateral_feats = []
        for i, feat in enumerate(dino_feats_2d):
            target_size = config.FEATURE_MAP_SIZES[str(i)]
            if feat.shape[-1] != target_size:
                feat = F.interpolate(feat, size=(target_size, target_size), mode="bilinear", align_corners=False)
            feat = self.lateral_convs[i](feat)
            lateral_feats.append(feat)

        # ============ 步骤4: 自顶向下融合（U-Net/FPN风格跳连） ============
        fused_feats = [None] * len(self.fpn_levels)
        top_idx = len(self.fpn_levels) - 1
        fused_feats[top_idx] = self.smooth_convs[top_idx](lateral_feats[top_idx])

        for i in range(top_idx - 1, -1, -1):
            upsampled = F.interpolate(
                fused_feats[i + 1],
                size=lateral_feats[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused_feats[i] = self.smooth_convs[i](lateral_feats[i] + upsampled)

        # ============ 步骤5: 返回 OrderedDict ============
        temporal_tokens = []
        if self.enable_temporal_attention:
            for i, feat in enumerate(fused_feats):
                refined, token = self.temporal_attn[i](feat)
                fused_feats[i] = refined
                temporal_tokens.append(token)

        self._last_temporal_tokens = temporal_tokens
        features = OrderedDict((str(i), fused_feats[i]) for i in range(len(self.fpn_levels)))
        
        return features


class TemporalFasterRCNNWrapper(nn.Module):
    """对 Faster R-CNN 增加时序一致性辅助损失。"""

    def __init__(
        self,
        detector: FasterRCNN,
        backbone: DinoV3BackboneWithAdapter,
        temporal_loss_weight: float,
        roi_temporal_fusion: TemporalROIFusion | None = None,
    ):
        super().__init__()
        self.detector = detector
        self.backbone_ref = backbone
        self.temporal_loss_weight = temporal_loss_weight
        self.roi_temporal_fusion = roi_temporal_fusion

    def _compute_temporal_consistency_loss(self):
        # 优先使用ROI proposal级时序损失，与论文主路径一致
        if self.roi_temporal_fusion is not None:
            roi_loss = self.roi_temporal_fusion.pop_last_temporal_loss()
            if roi_loss is not None:
                return roi_loss

        # 退化路径：保留旧的主干token一致性，避免禁用ROI时无辅助信号
        tokens = getattr(self.backbone_ref, "_last_temporal_tokens", [])
        if not tokens:
            return None

        loss = None
        valid_levels = 0
        for level_tokens in tokens:
            # level_tokens: [T, C]
            if level_tokens is None or level_tokens.shape[0] < 2:
                continue
            level_loss = F.mse_loss(level_tokens[1:], level_tokens[:-1])
            loss = level_loss if loss is None else loss + level_loss
            valid_levels += 1

        if valid_levels == 0:
            return None
        return loss / valid_levels

    def forward(self, images, targets=None):
        outputs = self.detector(images, targets)
        if self.training and isinstance(outputs, dict) and self.temporal_loss_weight > 0:
            temporal_loss = self._compute_temporal_consistency_loss()
            if temporal_loss is not None:
                outputs["loss_temporal_consistency"] = temporal_loss * self.temporal_loss_weight
        return outputs


# ======================== Faster R-CNN 模型构建 ========================
def build_faster_rcnn_model(num_classes: int = config.NUM_CLASSES):
    """
    构建完整的 Faster R-CNN 模型
    
    Args:
        num_classes: 类别数 (包括背景)
        
    Returns:
        model: Faster R-CNN 模型
    """
    print("=" * 50)
    print("构建 DINOv3 + Faster R-CNN 模型")
    print("=" * 50)
    
    # ============ 步骤1: 加载 DINOv3 预训练模型 ============
    print(f"加载 DINOv3 模型...")
    print(f"  仓库路径: {config.DINOV3_REPO_DIR}")
    print(f"  权重路径: {config.DINOV3_WEIGHTS_PATH}")

    _ensure_hub_cache_consistency(Path(config.DINOV3_WEIGHTS_PATH))
    
    try:
        # 使用 torch.hub.load 正确加载 DINOv3
        dino_model = torch.hub.load(
            str(config.DINOV3_REPO_DIR),  # 本地仓库路径
            'dinov3_vitb16',               # 模型名称
            source='local',                # 从本地加载
            weights=str(config.DINOV3_WEIGHTS_PATH)  # 权重路径
        )
        print("DINOv3 模型和权重加载成功！")
    
    except Exception as e:
        print(f"错误: 无法加载 DINOv3 模型")
        print(f"详细信息: {e}")
        print("\n请检查:")
        print(f"  1. 仓库路径是否正确: {config.DINOV3_REPO_DIR}")
        print(f"  2. 权重文件是否存在: {config.DINOV3_WEIGHTS_PATH}")
        raise RuntimeError("DINOv3 预训练权重加载失败，已中止训练。") from e
    
    # ============ 步骤2: 构建带 Adapter 的主干网络 ============
    print("\n构建 Adapter...")
    backbone = DinoV3BackboneWithAdapter(
        dino_model=dino_model,
        embed_dim=config.DINO_EMBED_DIM,
        out_channels=config.ADAPTER_OUT_CHANNELS,
        freeze_backbone=True,
        layer_indices=config.DINO_FPN_LAYER_INDICES,
    )
    
    # ============ 步骤3: 配置 RPN Anchor Generator ============
    # 为不同尺度的特征图设置不同的 anchor 大小
    anchor_sizes = ((32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # ============ 步骤4: 构建 Faster R-CNN ============
    print("\n构建 Faster R-CNN 检测头...")
    
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=None,  # 使用默认
        # 关键修改：显式设置 min_size 和 max_size，防止图像被resize
        min_size=config.IMAGE_SIZE,  # 1024
        max_size=config.IMAGE_SIZE,  # 1024
        # RPN 参数
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box Head 参数
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    )

    temporal_roi_fusion = None
    if config.ENABLE_ROI_TEMPORAL_FUSION:
        print("\n接入 ROI 级时序融合模块...")
        roi_feat_dim = model.roi_heads.box_predictor.cls_score.in_features
        temporal_roi_fusion = TemporalROIFusion(
            feat_dim=roi_feat_dim,
            num_heads=config.TEMPORAL_ATTENTION_HEADS,
            topk=config.ROI_TEMPORAL_TOPK,
            sim_thresh=config.ROI_TEMPORAL_SIM_THRESH,
        )
        model.roi_heads.box_head = TemporalBoxHead(model.roi_heads.box_head, temporal_roi_fusion)

        original_roi_forward = model.roi_heads.forward

        def forward_with_temporal_context(features, proposals, image_shapes, targets=None):
            proposal_counts = [p.shape[0] for p in proposals]
            temporal_roi_fusion.set_context(proposal_counts=proposal_counts, targets=targets)
            try:
                return original_roi_forward(features, proposals, image_shapes, targets)
            finally:
                temporal_roi_fusion.clear_context()

        model.roi_heads.forward = forward_with_temporal_context

    model = TemporalFasterRCNNWrapper(
        detector=model,
        backbone=backbone,
        temporal_loss_weight=config.TEMPORAL_LOSS_WEIGHT,
        roi_temporal_fusion=temporal_roi_fusion,
    )
    
    # ============ 步骤5: 确保只有 Adapter 和 Head 可训练 ============
    print("\n参数统计:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    print(f"  可训练比例: {trainable_params / total_params * 100:.2f}%")
    
    print("\n" + "=" * 50)
    print("模型构建完成！")
    print("=" * 50)
    
    return model


# ======================== 模型测试 ========================
if __name__ == "__main__":
    # 测试模型构建
    model = build_faster_rcnn_model(num_classes=2)
    
    # 测试前向传播
    print("\n" + "=" * 50)
    print("测试前向传播")
    print("=" * 50)
    
    # 创建虚拟输入
    dummy_images = [torch.randn(3, 1024, 1024) for _ in range(2)]
    dummy_targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[150, 150, 250, 250]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]
    
    # 训练模式 (返回 loss)
    model.train()
    try:
        loss_dict = model(dummy_images, dummy_targets)
        print("\n训练模式输出:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")
        print("\n前向传播测试成功！")
    except Exception as e:
        print(f"\n错误: {e}")
        raise
