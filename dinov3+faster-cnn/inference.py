# -*- coding: utf-8 -*-
"""
Baseline / Stage2 模型推理脚本

功能:
1. 支持加载 stage1(Baseline) 或 stage2(ROI temporal) checkpoint 做推理
2. 支持单图或文件夹批量推理
3. 保存预测结果到 JSON，可选保存可视化图片
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np

import config
from model import build_faster_rcnn_model


def build_model(checkpoint_path: Path, device: torch.device, enable_roi_temporal_fusion: bool):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"stage1 checkpoint 不存在: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    config.ENABLE_ROI_TEMPORAL_FUSION = enable_roi_temporal_fusion

    model = build_faster_rcnn_model(num_classes=config.NUM_CLASSES)
    load_result = model.load_state_dict(state_dict, strict=False)

    if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
        print("警告: checkpoint 与模型键存在不完全匹配")
        print(f"  missing keys: {len(load_result.missing_keys)}")
        print(f"  unexpected keys: {len(load_result.unexpected_keys)}")

    model.to(device)
    model.eval()
    return model


def _sort_case_ids(case_ids: List[str]) -> List[str]:
    def key_fn(case_id: str):
        try:
            return (0, int(case_id))
        except ValueError:
            return (1, case_id)

    return sorted(case_ids, key=key_fn)


def collect_images_by_case(input_path: Path) -> Dict[str, List[Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        return {"single": [input_path]}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if len(images) == 0:
        raise RuntimeError(f"目录中未找到图像文件: {input_path}")

    grouped: Dict[str, List[Path]] = {}
    for img_path in images:
        rel = img_path.relative_to(input_path)

        # 优先匹配 data08/<case_id>/images/*.jpg 的结构
        if len(rel.parts) >= 2 and rel.parts[1] == "images":
            case_id = rel.parts[0]
        elif len(rel.parts) >= 1:
            case_id = rel.parts[0]
        else:
            case_id = "unknown"

        grouped.setdefault(case_id, []).append(img_path)

    return {k: sorted(v) for k, v in grouped.items()}


def draw_boxes(image: Image.Image, boxes: List[List[float]], scores: List[float], labels: List[int], score_thr: float):
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 14)), f"cls={label} {score:.3f}", fill=(255, 0, 0))
    return image


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


def _name_to_category_id(name: str) -> int:
    n = str(name).strip().lower()
    for cid, cname in enumerate(config.CLASS_NAMES):
        if cid == 0:
            continue
        if n == str(cname).strip().lower():
            return cid
    # 单类任务兜底: 非背景统一映射到1
    return 1 if config.NUM_CLASSES > 1 else 0


def find_xml_for_image(img_path: Path) -> Path | None:
    # 优先匹配 data08/<case>/images/*.jpg 对应 data08/<case>/annotations/*.xml
    if img_path.parent.name == "images":
        xml_path = img_path.parent.parent / "annotations" / f"{img_path.stem}.xml"
        if xml_path.exists():
            return xml_path

    # 兜底: 同目录同名xml
    xml_path = img_path.with_suffix(".xml")
    if xml_path.exists():
        return xml_path

    return None


def parse_voc_xml(xml_path: Path) -> List[Dict]:
    root = ET.parse(xml_path).getroot()
    gts = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="Vascular Stenosis")
        cid = _name_to_category_id(name)
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin", default="0"))
            ymin = float(bbox.findtext("ymin", default="0"))
            xmax = float(bbox.findtext("xmax", default="0"))
            ymax = float(bbox.findtext("ymax", default="0"))
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        gts.append({"category_id": int(cid), "bbox": [xmin, ymin, xmax, ymax]})
    return gts


def evaluate_ap(preds, gt_by_image, class_ids: List[int], iou_thresh: float):
    ap_list = []

    for cls in class_ids:
        gt_cls = defaultdict(list)
        for image_id, anns in gt_by_image.items():
            boxes = [a["bbox"] for a in anns if int(a["category_id"]) == cls]
            gt_cls[image_id] = boxes

        num_gt = sum(len(v) for v in gt_cls.values())
        if num_gt == 0:
            continue

        dets = [p for p in preds if int(p["category_id"]) == cls]
        dets.sort(key=lambda x: x["score"], reverse=True)

        gt_used = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_cls.items()}
        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)

        for i, d in enumerate(dets):
            image_id = d["image_id"]
            pred_box = d["bbox"]

            gts = gt_cls.get(image_id, [])
            if len(gts) == 0:
                fp[i] = 1.0
                continue

            ious = [box_iou_xyxy(pred_box, g) for g in gts]
            best_idx = int(np.argmax(ious)) if len(ious) > 0 else -1
            best_iou = ious[best_idx] if best_idx >= 0 else 0.0

            if best_iou >= iou_thresh and not gt_used[image_id][best_idx]:
                tp[i] = 1.0
                gt_used[image_id][best_idx] = True
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


def evaluate_prf1_at_iou50(preds, gt_by_image, class_ids: List[int]):
    valid_preds = [p for p in preds if int(p["category_id"]) in class_ids]
    valid_preds.sort(key=lambda x: x["score"], reverse=True)

    gt_cls = defaultdict(list)
    for image_id, anns in gt_by_image.items():
        for a in anns:
            cid = int(a["category_id"])
            if cid in class_ids:
                gt_cls[image_id].append({"category_id": cid, "bbox": a["bbox"]})

    total_gt = sum(len(v) for v in gt_cls.values())
    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0

    gt_used = {image_id: np.zeros(len(anns), dtype=bool) for image_id, anns in gt_cls.items()}
    tp = np.zeros(len(valid_preds), dtype=np.float32)
    fp = np.zeros(len(valid_preds), dtype=np.float32)

    for i, d in enumerate(valid_preds):
        image_id = d["image_id"]
        cid = int(d["category_id"])
        pbox = d["bbox"]

        gts = gt_cls.get(image_id, [])
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

        if best_iou >= 0.5 and best_j >= 0 and not gt_used[image_id][best_j]:
            tp[i] = 1.0
            gt_used[image_id][best_j] = True
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


def main():
    parser = argparse.ArgumentParser(description="Baseline / Stage2 模型推理")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.CHECKPOINT_DIR / "base_stage1" / "best.pth"),
        help="模型 checkpoint 路径",
    )
    parser.add_argument("--input", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument("--output-dir", type=str, default="outputs/inference_stage2", help="推理结果输出目录")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="cuda 或 cpu")
    parser.add_argument("--score-thr", type=float, default=0.05, help="可视化与导出阈值")
    parser.add_argument(
        "--enable-roi-temporal-fusion",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="是否按第二阶段结构加载 ROI temporal fusion",
    )
    parser.add_argument("--save-vis", action="store_true", help="保存带框可视化图像")
    parser.add_argument("--eval", action="store_true", help="推理后计算 Prec/Rec/F1/mAP@0.5（需有XML标注）")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"ROI temporal fusion: {args.enable_roi_temporal_fusion}")

    model = build_model(checkpoint_path, device, args.enable_roi_temporal_fusion)

    images_by_case = collect_images_by_case(input_path)
    ordered_case_ids = _sort_case_ids(list(images_by_case.keys()))
    total_images = sum(len(images_by_case[cid]) for cid in ordered_case_ids)
    print(f"病例数: {len(ordered_case_ids)}")
    print(f"图像总数: {total_images}")

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    all_results: List[Dict] = []
    eval_preds: List[Dict] = []
    eval_gt_by_image = defaultdict(list)
    has_missing_xml = False
    done = 0

    for case_idx, case_id in enumerate(ordered_case_ids, 1):
        case_images = images_by_case[case_id]
        print(f"开始病例推理: {case_id} ({case_idx}/{len(ordered_case_ids)}), 图像数={len(case_images)}")

        for img_path in case_images:
            image = Image.open(img_path).convert("RGB")
            x = tfm(image).to(device)

            with torch.no_grad():
                out = model([x])[0]

            boxes = out["boxes"].detach().cpu().numpy().tolist()
            scores = out["scores"].detach().cpu().numpy().tolist()
            labels = out["labels"].detach().cpu().numpy().tolist()

            dets = []
            for b, s, c in zip(boxes, scores, labels):
                if float(s) < args.score_thr:
                    continue
                dets.append({
                    "bbox": [float(v) for v in b],
                    "score": float(s),
                    "category_id": int(c),
                })

            all_results.append(
                {
                    "case_id": case_id,
                    "image": str(img_path),
                    "num_dets": len(dets),
                    "detections": dets,
                }
            )

            if args.eval:
                image_key = str(img_path)
                xml_path = find_xml_for_image(img_path)
                if xml_path is None:
                    has_missing_xml = True
                else:
                    for gt in parse_voc_xml(xml_path):
                        eval_gt_by_image[image_key].append(gt)

                for det in dets:
                    eval_preds.append(
                        {
                            "image_id": image_key,
                            "category_id": int(det["category_id"]),
                            "bbox": det["bbox"],
                            "score": float(det["score"]),
                        }
                    )

            if args.save_vis:
                case_vis_dir = vis_dir / str(case_id)
                case_vis_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_boxes(image.copy(), boxes, scores, labels, args.score_thr)
                vis.save(case_vis_dir / f"{img_path.stem}_result.jpg")

            done += 1
            if done % 20 == 0 or done == total_images:
                print(f"推理总进度: {done}/{total_images}")

    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("推理完成")
    print(f"结果JSON: {json_path}")
    if args.save_vis:
        print(f"可视化目录: {vis_dir}")

    if args.eval:
        class_ids = list(range(1, config.NUM_CLASSES))
        if len(eval_gt_by_image) == 0:
            print("未找到可用XML标注，跳过指标计算。")
        else:
            if has_missing_xml:
                print("警告: 部分图像未找到对应XML标注，指标仅基于已找到标注的图像计算。")

            precision, recall, f1, best_conf = evaluate_prf1_at_iou50(eval_preds, eval_gt_by_image, class_ids)
            map_50 = evaluate_ap(eval_preds, eval_gt_by_image, class_ids, iou_thresh=0.5)

            print("\n========== 推理评估结果 ==========")
            print(f"Precision:      {precision:.6f}")
            print(f"Recall:         {recall:.6f}")
            print(f"F1:             {f1:.6f}")
            print(f"Best Conf(F1):  {best_conf:.6f}")
            print(f"mAP@0.5:        {map_50:.6f}")
            print("===================================")


if __name__ == "__main__":
    main()
