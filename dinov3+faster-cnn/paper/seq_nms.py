import numpy as np
import torch

# from utils.utils_seq import non_max_suppression, intersection_over_union, non_max_suppression_for_seq
from ultralytics.utils.utils_seq import intersection_over_union, non_max_suppression_for_seq
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy


class SeqNMS:
    def __init__(self,
                 link_conf_thres,
                 link_iou_thres,
                 nms_conf_thres,
                 nms_iou_thres,
                 rescore,
                 max_det=300,):
        self.link_conf_thres = link_conf_thres
        self.link_iou_thres = link_iou_thres
        self.nms_conf_thres = nms_conf_thres
        self.nms_iou_thres = nms_iou_thres
        assert rescore == 'ave' or rescore == 'max', "Use valid rescoring method"
        self.rescore = rescore
        self.max_det = max_det

        self.prev_bboxes = None
        self.prev_score = None
        self.prev_seq_length = None

    def _rm_low_conf(self, bboxes):
        """
        remove bboxes with low confidence

        :param bboxes: input bboxes without being filtered: [(batch)1, 5, num_bbox]
        :return: bboxes with higher confidence
        """
        mask = bboxes[0, -1, :] > self.link_conf_thres
        conf_bboxes = bboxes[:, :, mask]

        return conf_bboxes

    def _cal_bbox_iou(self, bbox):
        # prev_bbox: list[(x1, y1, x2, y2, conf, cls)...]
        # bbox: [5], [5.7480e+02, 4.5218e+02, 1.8730e+01, 1.7436e+01, 5.0115e-02
        ious = np.zeros(len(self.prev_bboxes))
        for i in range(len(self.prev_bboxes)):
            ious[i] = intersection_over_union(bbox[:4], self.prev_bboxes[i][:4], "corners").item()

        return ious

    def process(self, bboxes):
        """
        process a new bboxes

        :param bboxes: The input bboxes are in a list [1, 4+nc, number of boxes], the 4+nc dimension contains conf score,
        x center, y center, width and height of each bbox
        number of boxes example: 128*128 + 64*64 + 32*32
        :return:
        """
        assert bboxes.shape[0] == 1  # batch

        prev_bboxes = []
        prev_score = []
        prev_seq_length = []
        self.device = bboxes[0].device
        # nc = bboxes.shape[-1] - 4 - 1
        nc = bboxes.shape[-2] - 4

        bboxes = self._rm_low_conf(bboxes)
        # bboxes.shape: in/out[1, 5, num_bbox]

        if bboxes.shape[-1] == 0 or bboxes.shape[0] == 0:
            # 所有bbox都因低置信度被滤掉
            self.prev_bboxes = None
            self.prev_score = None
            self.prev_seq_length = None
            return [torch.zeros((0, 5+nc), device=self.device)]

        if self.prev_bboxes is None:
            # 我们首先开始处理第一帧：prev_bboxes=None
            nms_bboxes = non_max_suppression(bboxes, self.nms_conf_thres, self.nms_iou_thres, max_det=10)  # 进入的必须是xywh
            # len(bboxes) == batch, 此时必为1; bbox_i = (n, 6), 6: (x1, y1, x2, y2, conf, cls)

            for nms_bbox in nms_bboxes[0]:  # [0]将bboxes从列表拿出
                # each nms_bboxes will be used as the start of a sequence for the next frame
                # 每个经过nms的bbox将作为下一帧连接的开始
                prev_bboxes.append(nms_bbox)
                # confidence score of the current nms_bboxes after NMS
                # prev_score加入置信度分数
                prev_score.append(nms_bbox[-2])
                # we are dealing with the first frame, hence the sequence length for each bbox is 1
                # 加入首帧时长度为1
                prev_seq_length.append(1)

        else:
            # establish sequences
            # record the bbox ids from self.prev_bboxes that are linked to the bboxes at the current frame
            # len(bboxes) == batch = 1
            # bboxes.shape: [1, 5, num_bbox]
            bboxes_len = bboxes.shape[-1]
            link_id = -1 * np.ones(bboxes_len, dtype=np.int16)
            bboxes = bboxes.squeeze(0).permute(1, 0)  # [num_bbox, 5]
            bboxes[:, :4] = xywh2xyxy(bboxes[:, :4])

            for i in range(bboxes_len):
                bbox = bboxes[i]
                # bbox.shape: 5, 5.6544e+02, 4.4347e+02, 5.7480e+02, 4.5218e+02, 5.0115e-02
                # 计算当前帧的某个框和之前帧的有效框的交并比
                ious_bbox = self._cal_bbox_iou(bbox)
                # based on ious_bbox identify which bbox from the previous frame has the highest IOU with the bbox of
                # interest at the current frame. If this max IOU > predefined threshold, build a connection between
                # these two bboxes
                max_iou_id = np.argmax(ious_bbox)
                if ious_bbox[max_iou_id] > self.link_iou_thres:
                    link_id[i] = max_iou_id
                    # 如果当前帧与之前加入link的框相似，创建link: prev(n)->bbox_now(1)
                    # 过去帧的框可以连接多个现在帧的框
            # sequence selection and sequence rescoring
            # Note that each bbox from the current frame could be linked to the same bbox in the previous frame
            # from these multiple bboxes, we will only keep the one with the highest score
            # so that each bbox from the previous frame is only linked to at most 1 bbox at the current frame
            # after selecting the sequence, rescore each bbox according to either average or max
            current_score = bboxes[:, -1]
            bboxes_selected = []  # bboxes after selecting the appropriate sequences
            seq_length_selected = []  # record the sequence length which each bbox is part of
            conf_score_selected = []  # confidence score of the sequence which each bbox is part of
            # 之前，给现在帧找了唯一的从前帧，但一个从前帧可能对应多个现在帧
            # 这一段的目的是给每一个从前帧找一个唯一的现在帧的对应
            for prev_bbox_id in range(len(self.prev_bboxes)):
                # identify current bboxes that are linked to a specific bbox in the bbox list from the previous frame
                # 一个prevbbox连接多个现bbox, 而一个现bbox只被一个prev_bbox连接
                current_bboxes_to_prev_bbox = link_id == prev_bbox_id
                # 针对每一个prev，找到连接的现bbox
                # apply NOT
                current_bboxes_to_prev_bbox_ = np.logical_not(current_bboxes_to_prev_bbox)
                # change the score for bboxes not linked to the specific bbox from the previous frame to -1 to exclude them for highest score identification
                current_score_to_prev_bbox = current_score.clone()
                current_score_to_prev_bbox[current_bboxes_to_prev_bbox_] = -1
                # 将未成功连接的bbox的连接score变成-1
                # id of current bbox with the max score to the specified prev bbox
                max_bbox_id_to_prev_bbox = torch.argmax(current_score_to_prev_bbox)
                if current_score_to_prev_bbox[max_bbox_id_to_prev_bbox] == -1:
                    # 当前置信度最大的框和之前的框没有重叠，说明当前prev_bbox的连接全部断掉
                    continue

                selected_bbox = bboxes[max_bbox_id_to_prev_bbox]
                # 选择最大置信度的框进行连接
                # sequence length for the sequence corresponding to the current bbox + 1 to account for the current bbox
                seq_length_selected.append(self.prev_seq_length[prev_bbox_id] + 1)
                # 重塑bbox相近的score
                if self.rescore == 'max':
                    selected_bbox[-1] = torch.max(selected_bbox[-1], self.prev_score[prev_bbox_id])
                    # when we use max to rescore, we record the maximum score of a sequence
                    conf_score_selected.append(selected_bbox[-1])
                else:
                    # when we use avg to rescore, we record the average score of a sequence
                    conf_score_selected.append(selected_bbox[-1] + self.prev_score[prev_bbox_id])
                    # 还要改！！！！！
                    selected_bbox[-1] = conf_score_selected[-1] / seq_length_selected[-1]
                # 先将连接的bbox加入select
                bboxes_selected.append(selected_bbox)
            # For bboxes from the current frames that do not link to any bboxes from the previous frames
            # we add them all to bboxes_selected
            for i in range(len(link_id)):
                if link_id[i] == -1:
                    bboxes_selected.append(bboxes[i])
                    # 再将未连接但conf过关的bbox加入select
                    # store the conf score of bboxes that do not belong to any sequence for future rescoring
                    conf_score_selected.append(bboxes[i][-1].detach())
                    # sequence length for bboxes that do not belong to any sequence is 1
                    seq_length_selected.append(1)

            # perform nms on the bboxes that have been rescored
            bboxes_selected = torch.stack(bboxes_selected, dim=0)
            bboxes_selected = torch.concat([bboxes_selected, torch.tensor(seq_length_selected).reshape(-1, 1).to(self.device),
                                            torch.stack(conf_score_selected).reshape(-1, 1)], dim=1)

            # (conf, x, y, w, h, idx, conf_unchanged)

            # bboxes_selected = bboxes_selected.tolist()
            nms_bboxes = non_max_suppression_for_seq(bboxes_selected,
                                                     self.nms_iou_thres,
                                                     self.nms_conf_thres,
                                                     "corners",
                                                     max_det=10)
            for nms_bbox in nms_bboxes:
                prev_bboxes.append(torch.concat([nms_bbox[:5], torch.tensor([0], device=self.device)]))
                prev_seq_length.append(int(nms_bbox[5]))
                prev_score.append(nms_bbox[6])

        if len(prev_bboxes) == 0:
            # in case nms suppress all bbox, i.e., all bboxes have conf < threshold
            self.prev_bboxes = None
            self.prev_score = None
            self.prev_seq_length = None
            return [torch.zeros((0, 6), device=self.device)]

        self.prev_bboxes = prev_bboxes
        self.prev_score = prev_score
        self.prev_seq_length = prev_seq_length

        # return [torch.stack(prev_bboxes, dim=0)[:self.max_det]]
        return [torch.stack(prev_bboxes, dim=0)]
