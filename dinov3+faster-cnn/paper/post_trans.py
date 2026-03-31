import torch
import torch.nn as nn
import torchvision
from torchvision.ops.boxes import box_area

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors, TaskAlignedAssigner
try:
    from ultralytics.utils.atss import ATSSAssigner, generate_anchors
except Exception:  # ultralytics version compatibility: some versions do not provide ATSS utilities
    ATSSAssigner = None

    def generate_anchors(*args, **kwargs):
        raise ImportError("ultralytics.utils.atss is unavailable in the current environment")
import copy


class FeatureAggregationBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25, pos_embed=False, agg_reg=False):
        # dim: input[batch, sequence length, input dimension] --> output[batch, sequence length, dim]
        # 输入：[1, b*topK, channels: 256]
        super().__init__()
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if self.pos_embed:
            self.qkv_cls = nn.Linear(dim+2, dim * 3, bias=qkv_bias)
        else:
            self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(64, 64 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.agg_reg = agg_reg

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,
                mask=True, sim_thresh=0.75,
                use_mask=True, **kwargs):
        sim_mask, obj_mask = None, None
        B, N, C_cls = x_cls.shape  # (1, 106, 64)
        _, _, C_reg = x_reg.shape
        if self.pos_embed:
            C_cls -= 2
            C_reg -= 2
        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C_cls // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # (1, 8, 106, 64/8)
        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)
        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        if self.agg_reg:
            qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C_reg // self.num_heads).permute(2, 0, 3, 1, 4)
            q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]
            q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
            k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
            v_reg_normed = v_reg / torch.norm(v_reg, dim=-1, keepdim=True)
            attn_reg_raw = v_reg_normed @ v_reg_normed.transpose(-2, -1)

        if use_mask:
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        if self.agg_reg:
            attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale  # * fg_score * fg_score_mask
            attn_reg = attn_reg.softmax(dim=-1)
            attn_reg = self.attn_drop(attn_reg)

        attn = (attn_cls + attn_reg) / 2 if self.agg_reg else attn_cls  # all_attn: [1, 8, n_p, n_p]
        
        if mask:
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')
            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False) / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)  # (b, n_p, n_p)
            sim_mask = sim_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # # (b, n_h, n_p, n_p)
            if self.agg_reg:
                attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False)[0] / self.num_heads
                obj_mask = torch.where(attn_reg_raw > sim_thresh, ones_matrix, zero_matrix)
            if use_mask:
                sim_mask = sim_mask * cls_score_mask[0, 0, :, :] * fg_score_mask[0, 0, :, :]
            attn = sim_mask * attn
            if self.agg_reg:
                obj_mask = obj_mask * sim_mask / (torch.sum(obj_mask * sim_mask, dim=-1, keepdim=True))
                attn = obj_mask * attn

        # v_cls: [1, num_head, b*topK, c//n_h]
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C_cls)
        x_cls = torch.cat([x, x_cls], dim=-1)

        if self.agg_reg:
            x_post_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N, C_reg)
            x_reg = torch.cat([x_post_reg, x_reg], dim=-1)

        return x_cls, x_reg, sim_mask, obj_mask


class MSA_yolov(nn.Module):
    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False, pos_embed=False):
        super().__init__()
        self.reconf = reconf
        self.fab = FeatureAggregationBlock(dim, num_heads, qkv_bias, attn_drop, scale=scale, pos_embed=pos_embed)
        self.linear = nn.Linear(2 * dim, out_dim)
        if reconf:
            self.linear1_obj = nn.Linear(2 * dim, out_dim)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False, **kwargs):
        trans_cls, trans_obj, ave_mask, obj_mask = self.fab(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave,
                                                            use_mask=use_mask, **kwargs)
        trans_cls = self.linear(trans_cls)
        if self.reconf:
            trans_obj = self.linear_obj(trans_obj)
        return trans_cls, trans_obj


class YOLOVLoss(v8DetectionLoss):  # 输入：nn/task: 443
    def __init__(
        self,
        model,
        num_classes=1,
        width=1.0,
        #strides=(4, 8, 16, 32),
        strides=(8, 16, 32),
        in_channels=(256, 512, 1024),
        act="silu",
        depthwise=False,
        heads=4,
        drop=0.0,
        use_score=True,
        defualt_p=30,
        pre_nms=0.7,  # nms的iou阈值
        ave=True,
        defulat_pre=750,
        test_conf=0.001,
        use_mask=False,
        gmode=True,
        lmode=False,
        both_mode=False,
        reassign=False,
        pos_embed=False,
        localBlocks=1,
        **kwargs
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__(model=model)
        self.Afternum = defualt_p
        self.Prenum = defulat_pre
        self.simN = defualt_p
        self.nms_thresh = pre_nms
        self.n_anchors = 1
        self.use_score = use_score
        # self.num_classes = num_classes
        self.num_classes = self.nc
        self.decode_in_inference = True  # for deploy, set to False
        self.gmode = gmode
        self.lmode = lmode
        self.both_mode = both_mode

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        self.width = int(64 * width) + 2 * int(pos_embed)
        self.pos_embed = pos_embed
        self.ave = ave
        self.use_mask = use_mask
        self.training = True
        self.reassign = reassign

        self.max_rp = kwargs.get("max_rp", 15)
        self.min_rp = kwargs.get("min_rp", 1)
        self.sim_thresh = kwargs.pop("sim_thresh", 0.9)
        drop = model.args["dropout"] if isinstance(model.args, dict) else model.args.dropout

        # self.agg = MSA_yolov(dim=self.width, out_dim=4 * self.width,
        #                      num_heads=heads, attn_drop=drop, reconf=kwargs.get('reconf',False))
        # self.cls_pred = nn.Linear(4*self.width, num_classes)
        self.agg = model.model[-2]  # nn/task: 1047
        self.cls_pred = model.model[-1]
        self.assigner_vid = TaskAlignedAssigner(topk=self.max_rp, num_classes=self.nc, alpha=0.5, beta=6.0)
        # self.assigner_vid = ATSSAssigner(self.max_rp, num_classes=self.nc)

        if both_mode:
            self.g2l = nn.Linear(int(4 * self.width), self.width)

        self.stems = nn.ModuleList()
        self.kwargs = kwargs

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        # self.iou_loss = IOUloss(reduction="none")
        self.stride = strides
        self.ota_mode = kwargs.get('ota_mode', False)
        self.grids = [torch.zeros(1)] * len(in_channels)

    # def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5, lframe=0, gframe=32):
    def __call__(self, preds, batch, **kwargs):
        # 输入的pred:
        # train: x, vid_feat, 分别是两个长度3的列表，代表三种特征图
        # x, y, vid_feat, y为decode_res: [10, 5, 21504]
        self.training = kwargs.get("training", True)
        raw_preds = preds
        if isinstance(raw_preds, tuple) and len(raw_preds) == 2 and isinstance(raw_preds[1], dict):
            raw_preds = raw_preds[1]

        if isinstance(raw_preds, dict):
            # Ultralytics>=8.3 detect head may return dict outputs in training.
            det_preds = raw_preds.get("one2many", raw_preds)
            if not isinstance(det_preds, dict):
                raise TypeError(f"Unexpected one2many prediction type: {type(det_preds)}")
            required_keys = {"boxes", "scores", "feats"}
            if not required_keys.issubset(det_preds):
                raise KeyError(f"Prediction dict missing keys {required_keys}, got {set(det_preds.keys())}")

            preds = det_preds["feats"]
            decode_res = torch.cat([det_preds["boxes"], det_preds["scores"].sigmoid()], dim=1)
            vid_features = preds
        else:
            if not isinstance(raw_preds, (list, tuple)) or len(raw_preds) != 3:
                raise TypeError(f"Unexpected prediction container: {type(raw_preds)} with len={len(raw_preds) if hasattr(raw_preds, '__len__') else 'NA'}")
            preds, decode_res, vid_features = raw_preds  # decode_res: [10, 5, 21504]

        if not isinstance(decode_res, torch.Tensor):
            raise TypeError(f"decode_res must be Tensor, got {type(decode_res)}")
        decode_res = torch.permute(decode_res, (0, 2, 1))

        imgs = batch["img"]
        outputs = []
        batch_size = imgs.shape[0]
        loss_v8 = torch.zeros(4, device=self.device)

        # feats = preds[1] if isinstance(preds, tuple) else preds  # tuple: feat_img/feat_vect
        # anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # pos_embed = (anchor_points / stride_tensor * 2 - 1).repeat(10, 1, 1)  # 128/64/32

        feats = preds[1] if isinstance(preds, tuple) else preds  # tuple: feat_img/feat_vect
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # Use runtime batch size so sequence batches do not depend on fixed 10-frame preprocessing.
        frame_count = imgs.shape[0]
        pos_embed = (anchor_points / stride_tensor * 2 - 1).repeat(frame_count, 1, 1)
        if self.training:
            loss_v8 = self.getloss(preds, batch)
            # feats = preds[1] if isinstance(preds, tuple) else preds  # tuple: feat_img/feat_vect
            # anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
            # pos_embed = (anchor_points / stride_tensor * 2 - 1).repeat(10, 1, 1)  # 128/64/32

            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1
            )  # feats[0].shape[0] == batch, self.no = 65 = 16*4+1
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()
            # pred_distri为框点离锚框中心距离的分布: (batch, 8400, 64)
            # pred_scores为框点的分值: (batch, 8400, 1)

            dtype = pred_scores.dtype
            # batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

            # anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
            reg_outputs = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, 8400, 4)

            outputs = torch.cat([reg_outputs * stride_tensor, pred_scores], dim=-1)
            # pred_scores不是0-1
            # 注意reg_outputs还只是在特征图上的坐标，还原到原图还得*stride

            # decode_res = torch.cat([reg_outputs, pred_scores.sigmoid()], dim=-1)
            # outputs本应该是列表，

            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            # 假如target有10个tumor，则一行数据为:(肿瘤所属图片号, 类别0, center/w/h)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            # out.shape: (batch, max_count, 1cls+4xyxy)
            # 这个就是下面get_fg_idx的target
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            # gt_bboxes: (batch, max_count, 4)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
            # mask_gt: (batch, max_count)
            num_gts = mask_gt.sum()

            if ATSSAssigner is not None and isinstance(self.assigner, ATSSAssigner):
                anchors, _, n_anchors_list, _ = \
                    generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                                     device=feats[0].device)
                target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                    anchors,
                    n_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                    reg_outputs.detach() * stride_tensor)
            else:
                target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                    pred_scores.detach().sigmoid(),
                    (reg_outputs.detach() * stride_tensor).type(gt_bboxes.dtype),
                    anchor_points * stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                )
            # target_labels: [8, 8400]
            # target_bboxes:[8, 8400, 4], target_scores:[8, 8400, nc], fg_mask.shape:[8, 8400]
            # target_gt_idx: 每个preds对应的gt的序号
            # fg_mask.shape: [8, 8400], target_gt_idx:[8, 8400]

        # decode_res: (b, h*w, 4+nc), 4:xywh; 1:sigmoid
        pred_result, agg_idx = self.postprocess_widx(decode_res,
                                                     num_classes=self.num_classes,
                                                     nms_thre=self.nms_thresh)

        raw_reg_features = [vid_feature[:, :64, :, :] for vid_feature in vid_features]
        raw_cls_features = [vid_feature[:, 64:, :, :] for vid_feature in vid_features]

        cls_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in raw_cls_features], dim=2
        ).permute(0, 2, 1)  # [b,features,channels]: [b, 8400, 64]
        reg_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in raw_reg_features], dim=2
        ).permute(0, 2, 1)

        if self.pos_embed:
            cls_feat_flatten = torch.concat([pos_embed, cls_feat_flatten], dim=-1)

        # ota_idxs是前景索引
        # decode_res: [batch,feature_num,5+clsnum]
        # pred_result: [batch,topK,5+cls]
        # agg_idx/output_index: list:[n], n为剩下的idx, len==batch
        preds_per_frame = []
        for p in agg_idx:
            if p is None: preds_per_frame.append(0)
            else: preds_per_frame.append(p.shape[0])
        # preds_per_frame: agg_idx在每张图的数量

        if sum(preds_per_frame) == 0 and self.training:
            loss_v8[-1] = torch.tensor(0).to(self.device)

            loss_v8[0] *= self.hyp.box  # box gain
            loss_v8[1] *= self.hyp.cls  # cls gain
            loss_v8[2] *= self.hyp.dfl  # dfl gain

            # return loss_v8.sum() * batch_size, loss_v8.detach()
            return loss_v8[-1] * batch_size, loss_v8.detach()
        # loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1),
        # loss_refined_cls, reg_weight * loss_refined_iou, loss_refined_obj,

        # 输入的cls_feat_flatten是未卷积的原始特征：(b, 8400, 256)
        # 而输入的pred_result已经是: [batch, topK, 5+1],
        # 输出的features_cls: (k_top, 8400, 256)
        (features_cls, features_reg, cls_scores, locs) = \
            self.find_feature_score(cls_feat_flatten, agg_idx, reg_feat_flatten, pred_result)
        # return features_cls, features_reg, cls_scores, fg_scores, locs, all_scores
        # pred_result: list,len==b, (K, 5+1)
        # feature_cls, features_reg: (b*topK, 256)  # b->series, 相当于一个series里所有的有效特征
        # if features_cls == None and not self.training: return pred_result, pred_result
        # if features_cls.shape[0] == 0 and not self.training: return pred_result, pred_result
        # locs: [b*topK, 4], 实际的pred位置

        features_reg_raw = features_reg.unsqueeze(0)
        features_cls_raw = features_cls.unsqueeze(0)  # [1, b*topK, channels]

        # [1, series, features(topK), channels]

        cls_scores = cls_scores.to(cls_feat_flatten.dtype)
        # fg_scores = fg_scores.to(cls_feat_flatten.dtype)
        locs = locs.to(cls_feat_flatten.dtype)
        locs = locs.view(1, -1, 4)

        more_args = {'width': imgs.shape[-1],
                     'height': imgs.shape[-2],
                     'fg_score': None,
                     'cls_score': cls_scores,
                     'afternum': self.Afternum,
                     'use_score': self.use_score}

        if self.kwargs.get('agg_type', 'fab') == 'fab':
            kwargs = self.kwargs
            # self.agg = self.agg.to(dtype)
            kwargs.update({#'lframe': lframe,
                           #'gframe': gframe,
                           'afternum': self.Afternum})
            features_cls, features_reg = self.agg(features_cls_raw, features_reg_raw, cls_scores, None, # his: fg_scores
                                                  sim_thresh=self.sim_thresh,
                                                  ave=self.ave, use_mask=self.use_mask, **kwargs)
            cls_preds = self.cls_pred(features_cls)  # (b*topK, 1024)->(b*topK, num_class/1)

            if self.kwargs.get('reconf', False):
                # self.obj_pred = self.obj_pred.to(dtype)
                obj_preds = self.obj_pred(features_reg)
                # 预测质量obj，和reg共享一个解耦头，v8没有这个字段，(b*topK, 1024)->(b*topK, 1)
                reg_preds = None
            else:
                obj_preds, reg_preds = None, None

        # training
        # outputs = torch.cat(outputs, 1)  # 已经是[8, 21504, 5]了
        # [batch, n_anchors_all, 4+1+nc]
        if self.training:
            cls_targets = [label[agg_idx[i]] for i, label in enumerate(target_scores)]
            reg_targets = [bbox[agg_idx[i]] for i, bbox in enumerate(target_bboxes)]

            cls_targets = torch.cat(cls_targets, 0)
            reg_targets = torch.cat(reg_targets, 0)

            # obj_targets = torch.cat(obj_targets, 0)
            # fg_masks = torch.cat(fg_masks, 0)
            num_fg = sum([len(idx) for idx in agg_idx])
            # cls_targets是sigmoid后的，而cls_preds不是，这是BCEWithLogitsLoss的要求
            (ref_loss, ref_loss_item) = self.get_losses_refine(
                fg_mask,
                num_fg,
                num_gts,
                None,  # l1_targets
                None,
                cls_preds,
                reg_preds,
                # refine_obj_masks,  # 无关
                # refined_cls_targets=refine_cls_targets if self.reassign else cls_targets,
                # refined_cls_masks=refine_cls_masks if self.reassign else None,
                refined_cls_targets=cls_targets,
                refined_cls_masks=None
                # None,  # 无关
            )

            loss_v8[-1] = ref_loss * 5
            # loss_v8[-1] = ref_loss * 1
            loss_v8[0] *= self.hyp.box  # box gain
            loss_v8[1] *= self.hyp.cls  # cls gain
            loss_v8[2] *= self.hyp.dfl  # dfl gain
            return loss_v8.sum() * batch_size, loss_v8.detach()

        else:
            cls_per_frame, obj_per_frame, reg_per_frame = [], [], []
            for i in range(len(preds_per_frame)):
                if self.kwargs.get('reconf', False):
                    obj_per_frame.append(obj_preds[:preds_per_frame[i]].squeeze(-1))
                    obj_preds = obj_preds[preds_per_frame[i]:]

                cls_per_frame.append(cls_preds[:preds_per_frame[i]])
                cls_preds = cls_preds[preds_per_frame[i]:]
                # cls_preds是一个batch里的：(b*topK, nc)
                # 每次取出一部分cls_pred到cls_per_frame: (topK, nc)

            if not self.kwargs.get('reconf', False):
                obj_per_frame = None
            result, result_ori = postprocess(copy.deepcopy(pred_result),
                                             self.num_classes,
                                             cls_per_frame,
                                             conf_output=obj_per_frame,
                                             nms_thre=0.1,
                                             )
            return result, decode_res

    def find_feature_score(self, features, idxs, reg_features, predictions=None):
        # cls_feat_flatten,
        # agg_idx,
        # reg_feat_flatten,
        # pred_result
        # 根据index(每张图的topK的特征索引)从flatten后的特征图里拿出对应的cls和reg
        # 问：prediction里不是有吗
        # features: (b, 8400, 256)
        # predictions: (b, top_K, 5+1)
        features_cls = []
        # features_cls元素：(K, 256)
        features_reg = []
        cls_scores = []
        # all_scores = []
        fg_scores = []
        locs = []
        for i, feature in enumerate(features):  # len(features) == batch
            if idxs[i] is None or idxs[i] == []:
                continue

            features_cls.append(feature[idxs[i]])
            features_reg.append(reg_features[i, idxs[i]])
            cls_scores.append(predictions[i][:, 4])  # history: 5
            # fg_scores.append(predictions[i][:, 4])  # predictions[i][:, 4]: (topK)
            locs.append(predictions[i][:, :4])
            # all_scores.append(predictions[i][:, -self.num_classes:])
        if len(features_cls) == 0:
            # without any preds
            return None, None, None, None

        features_cls = torch.cat(features_cls)
        features_reg = torch.cat(features_reg)
        cls_scores = torch.cat(cls_scores)
        locs = torch.cat(locs)

        # feature_cls: (b * topK, 256)
        return features_cls, features_reg, cls_scores, locs

    def get_losses_refine(
            self,
            fg_masks,
            num_fg,
            num_gts,
            l1_targets,  # None
            origin_preds,  # None
            refined_cls,  # 经过交互优化后的cls_pred: (b*topK, nc)
            refined_reg,
            refined_cls_targets,  # 经过交互优化后的pred作loss的target
            refined_cls_masks,
    ):
        # if refined_obj_targets == None:
        #     pass
        # refined_obj_targets = refined_obj_masks.type_as(refined_obj)
        # refined_obj_targets = refined_obj_targets.view(-1, 1)
        # refined_obj_masks = refined_obj_masks.bool().squeeze(-1)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        loss_refined_iou = 0  # 已废弃
        if refined_cls_masks is not None:
            refined_cls_fg = max(float(torch.sum(refined_cls_masks)), 1)
            loss_refined_cls = (
                                   self.bcewithlog_loss(
                                       refined_cls.view(-1, self.num_classes)[refined_cls_masks],
                                       refined_cls_targets[refined_cls_masks]
                                   )
                               ).sum() / refined_cls_fg
        else:
            loss_refined_cls = (
                                   self.bcewithlog_loss(
                                       refined_cls.view(-1, self.num_classes), refined_cls_targets
                                   )
                               ).sum() / max(refined_cls_targets.shape[0], 1)

        reg_weight = 3.0
        loss = loss_refined_cls + reg_weight * loss_refined_iou # + loss_refined_obj

        return loss, (loss_refined_cls, loss_refined_iou)

    def postprocess_widx(self, prediction, num_classes, nms_thre=0.5, conf_thresh=0.001):
        # find topK predictions, play the same role as RPN
        # ota_idxs：前景mask，(batch, n)
        '''
        Args:
            prediction: [batch,feature_num,5+clsnum], 改：(b, h*w, 4+1), 4:xywh; 1:sigmoid
            num_classes:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        '''
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [torch.zeros((0, 6+num_classes), device=self.device) for _ in range(len(prediction))]
        output_index = [torch.zeros((0, 6+num_classes), device=self.device) for _ in range(len(prediction))]
        # reorder_cls = [None for _ in range(len(prediction))]
        # refined_obj_masks = []
        for i, image_pred in enumerate(prediction):
            # take ota idxs as output in training mode
            # obj_mask = torch.zeros(0,1)
            if not image_pred.size(0):
                continue
            # Get score and class with the highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 4: 4 + num_classes], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            # Now input: (c)
            detections = torch.cat(
                (image_pred[:, :4], class_conf, class_pred.float(), image_pred[:, 4: 4 + num_classes]), 1)
            # detection中原本只有reg、obj和class的可能性，在中间插入类的置信度和对应类别
            # detection: (x1, y1, x2, y2, class_conf_max, class_pred, class_conf_all)

            # detection: (8400, 4+1+2+nc)
            conf_mask = (detections[:, 4] >= conf_thresh).squeeze()  # class_conf_max
            # conf_mask: (8400)
            minimal_limit = self.min_rp
            maximal_limit = self.max_rp
            # 最少选择数和最多选择数量
            if minimal_limit != 0:
                # add a minimum limitation to the number of detections
                # >= conf_thresh不足则用更多topk进行补充
                if conf_mask.sum() < minimal_limit:
                    # get top minimal_limit detections
                    _, top_idx = torch.topk(detections[:, 4], minimal_limit)
                    conf_mask[top_idx] = True
            if maximal_limit != 0:
                # add a maximum limitation to the number of detections
                if conf_mask.sum() > maximal_limit:
                    # >= conf_thresh超过则重定（提高）阈值，使得选出的数量为maximal_limit
                    _, top_idx = torch.topk(detections[:, 4], maximal_limit)
                    conf_mask = torch.zeros_like(conf_mask)
                    conf_mask[top_idx] = True

            conf_idx = torch.where(conf_mask)[0]  # out: [1, 8400]->[8400]
            detections = detections[conf_mask]
            # if not detections.size(0):
            #     refined_obj_masks.append(obj_mask)
            #     continue
            if self.kwargs.get('use_pre_nms', True):
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4],
                    detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torch.arange(detections.shape[0])

            abs_idx_out_ota = conf_idx[nms_out_index]  # conf_idx是置信度过滤
            abs_idx = abs_idx_out_ota.cpu()
            bg_mask = torch.zeros_like(abs_idx_out_ota).cpu()
            # obj_mask = torch.cat((obj_mask.type_as(bg_mask), bg_mask.unsqueeze(1)))

            detections = detections[nms_out_index]
            # detection: (8400, 4+1+2+nc)->(nf, 4+1+2+nc)

            output[i] = torch.cat((output[i], detections))
            # output[i]: output[i] = detections[ota_idx, :]
            # ->(nf+ota_idx, 4+1+2+nc)

            if len(output_index[i]) == 0:
                if self.kwargs.get('use_pre_nms', True):
                    output_index[i] = conf_idx[nms_out_index]
                else:
                    output_index[i] = abs_idx
            else:
                # if abs_idx.shape[0] != 0:
                output_index[i] = torch.cat((output_index[i], abs_idx))

        return output, output_index

    def get_iou_based_label(self, pred_result, idx, labels, outputs, reg_targets, cls_targets):
        # pred_result, agg_idx, targets, outputs, target_bboxes, cls_targets
        # idx: agg_idx: list,len==b, [(k)], 指向pred_result
        # reg_targets: # (8, 8400, 4)
        # cls_targets: # (8, 8400)
        # labels: (b, max_count, 1cls+4bbox)
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
        # 存疑，看看labels如何初始化的
        # 这里label传入只为计数，我们之后用fg_mask加和后传入再用吧
        refine_cls_targets = []
        refine_cls_masks = []
        # refine_obj_targets = []
        # refine_obj_masks = []
        for batch_idx in range(len(pred_result)):
            num_gt = int(nlabel[batch_idx])
            gt_xyxy = reg_targets[batch_idx]
            if idx[batch_idx] is None: continue
            if num_gt == 0:
                #TODO: handle condition when idx[batch_idx] is None
                refine_cls_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                refine_cls_target[:, -1] = 1  # set no supervision to 1 as flag
                # refine_obj_target = outputs.new_zeros((idx[batch_idx].shape[0], 2))
            else:
                refine_cls_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                # refine_obj_target = outputs.new_zeros((idx[batch_idx].shape[0], 2))

                # gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))， 不用变，reg_target已是xyxy
                pred_box = pred_result[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)
                # max_iou找的是每个pred_bbox对应的最合适的gt 的iou
                # 出问题了,iou的维度为0：不太可能啊，num_gt==0的情况不是考虑了吗
                # pred_box为0
                cls_target = cls_targets[batch_idx]

                # refine_obj_target[:, -1] = 1  # set no supervision to 1 as flag, 后续变成0才保留
                # refine_obj_target = refine_obj_target.type_as(gt_xyxy)
                refine_cls_target[:, -1] = 1  # set no supervision to 1 as flag
                refine_cls_target = refine_cls_target.type_as(cls_target)

                fg_cls_coord = torch.where(max_iou.values >= 0.6)[0]
                bg_coord = torch.where(max_iou.values < 0.3)[0]
                fg_cls_max_idx = max_iou.indices[fg_cls_coord]
                cls_target_onehot = (cls_target > 0).type_as(cls_target)

                fg_ious = max_iou.values[fg_cls_coord].unsqueeze(-1)
                fg_ious = fg_ious.type_as(cls_target)
                refine_cls_target[fg_cls_coord, :self.num_classes] = cls_target_onehot[fg_cls_max_idx, :] * fg_ious
                # cls_target_onehot: [13, 1]
                refine_cls_target[fg_cls_coord, -1] = 0

            refine_cls_targets.append(refine_cls_target[:, :-1])
            # refine_obj_targets.append(refine_obj_target[:, :-1])
            refine_cls_masks.append(refine_cls_target[:, -1] == 0)
            # refine_obj_masks.append(refine_obj_target[:, -1] == 0)

        return refine_cls_targets, refine_cls_masks  # , refine_obj_targets, refine_obj_masks


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area, iou


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def postprocess(prediction, num_classes, fc_outputs,
                conf_output, conf_thre=0.001, nms_thre=0.5,
                cls_sig=True, return_idx=False):
    # fc_outputs: msa的cls挑选后的，相当于优化了的cls评分, list:(n, nc)
    # conf_output: None
    # prediction: 原: 4+obj+conf+pred+nc
    # 现: 4+nc, 目标：4+conf+pred+nc
    device = prediction[0].device
    output = [torch.zeros((0, 6), device=device) for _ in range(len(prediction))]
    output_ori = [torch.zeros((0, 6), device=device) for _ in range(len(prediction))]
    prediction_ori = copy.deepcopy(prediction)
    cls_pred, cls_conf = [], []
    for _ in range(len(prediction)):
        tmp_cls, tmp_pred = torch.max(fc_outputs[_], -1, keepdim=False)
        cls_pred.append(tmp_pred)
        cls_conf.append(tmp_cls)
    # cls_conf, cls_pred = torch.max(fc_outputs, -1, keepdim=False) #
    nms_out_idxs = []
    for i, detections in enumerate(prediction):
        if detections == None or not detections.size(0):
            continue
        if conf_output is not None:
            detections[:, 4] = conf_output[i].sigmoid()

        # detections[:, 5] = cls_conf[i].sigmoid()
        # detections[:, 6] = cls_pred[i]

        detections = torch.concat([detections[:, :4],
                                   cls_conf[i].sigmoid().unsqueeze(-1),
                                   cls_pred[i].unsqueeze(-1),
                                   detections[:, 4:]], dim=-1)
        # 目前：4+2+1: reg + conf + pred + nc

        if cls_sig:
            tmp_cls_score = fc_outputs[i].sigmoid()
        else:
            tmp_cls_score = fc_outputs[i]

        # tmp_cls_score: (n, nc)
        cls_mask = tmp_cls_score >= conf_thre
        cls_loc = torch.where(cls_mask)
        scores = torch.gather(tmp_cls_score[cls_loc[0]],  # 只有到阈值的
                              dim=-1,
                              index=cls_loc[1].unsqueeze(1))
        # score: (n-no, 1)
        #[:,cls_loc[1]]#tmp_cls_score[torch.stack(cls_loc).T]#torch.gather(tmp_cls_score, dim=1, index=torch.stack(cls_loc).T)

        detections[:, -num_classes:] = tmp_cls_score
        # 重新根据新cls标记各个cls的可能性
        # 原input: reg4, obj, class_conf, pred, nc
        # 现：reg4, class_conf, pred, nc

        # 用修正后的cls取代原cls预测
        detections_raw = detections[:, :6]
        # 原：4+1+2, reg + obj + cls_prob + cls
        # 现：4+2, reg + cls_prob + cls
        new_detections = detections_raw[cls_loc[0]]
        # 如果有多个class分类并且都过阈值，则该点会重复计数
        new_detections[:, -1] = cls_loc[1]  # 可能性最大的cls
        new_detections[:, 4] = scores.squeeze()
        detections_high = new_detections  # new_detections

        conf_mask = (detections_high[:, 4] >= conf_thre).squeeze()
        detections_high = detections_high[conf_mask]

        if not detections_high.shape[0]:
            continue
        if len(detections_high.shape) == 3:
            detections_high = detections_high[0]
        nms_out_index = torchvision.ops.batched_nms(
            detections_high[:, :4],
            detections_high[:, 4],
            detections_high[:, 5],
            nms_thre,
        )

        detections_high = detections_high[nms_out_index]
        output[i] = detections_high

        detections_ori = prediction_ori[i]  # 4+1
        # detections_ori = detections_ori[:, :7]
        conf_mask = detections_ori[:, 4] >= conf_thre
        # conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
        detections_ori = detections_ori[conf_mask]

        nms_out_index = torchvision.ops.batched_nms(
            detections_ori[:, :4],
            detections_ori[:, 4],   # conf
            detections_ori[:, 5],  # class, 直接就全0吧，咱们就一类
            nms_thre,
        )

        detections_ori = detections_ori[nms_out_index]
        output_ori[i] = detections_ori
        if return_idx:
            nms_out_idxs.append(nms_out_index + i*detections.size(0))
    if return_idx:
        return output, output_ori, torch.cat(nms_out_idxs, dim=0)
    return output, output_ori
