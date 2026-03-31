import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ========= 需要你修改的配置 =========
JSON_PATH   = r"D:\Study\VascularStenosis\CNN+LSTM\dataset\data02\data02_structure.json"  # 你的JSON路径
BASE_DATA02 = r"D:\Study\VascularStenosis\CNN+LSTM\dataset\data02"                # data02 目录的绝对路径
OUT_DIR     = r"D:\Study\VascularStenosis\CNN+LSTM\dataset\yolo_dataset"          # 输出YOLO数据集根目录
CLASSES     = ["stenosis"]  # YOLO类别名列表；我们把所有object都当成这个类别
# 只保留特定类名（来自XML <name>），留空或None表示不过滤（所有object都保留）
ALLOWED_LABELS = None  # 例如：{"stenosis", "lesion"}；默认不过滤

# 划分模式：'auto' 按病例自动8/2划分；或 'manual' 手动指定病例列表
SPLIT_MODE = 'auto'
TRAIN_CASES = []  # SPLIT_MODE='manual'时，填例如 ['200','201',...]
VAL_CASES   = []  # 同上
# ===================================


def resolve_path(base_data02: Path, p_str: str) -> Path:
    """
    将 JSON 中的路径（开头可能包含 'data02\\'）转换为绝对路径：BASE_DATA02 / 相对路径
    """
    p_norm = p_str.replace("/", os.sep).replace("\\", os.sep)
    parts = Path(p_norm).parts
    # 如果第一个片段是data02，去掉它
    if len(parts) > 0 and parts[0].lower() == "data02":
        p_rel = Path(*parts[1:])
    else:
        p_rel = Path(p_norm)
    return base_data02 / p_rel


def parse_voc_xml_to_yolo(xml_file: Path, img_w: int, img_h: int, allowed_labels=None):
    """
    解析单个VOC XML，输出YOLO标签行列表：[ "class_id x_center y_center width height", ... ]
    坐标归一化到[0,1]。allowed_labels为集合时，仅保留此集合中的<name>。
    """
    lines = []
    try:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
    except Exception as e:
        print(f"[WARN] 解析XML失败: {xml_file} -> {e}")
        return lines

    for obj in root.findall("object"):
        name_node = obj.find("name")
        cls_name = name_node.text.strip() if name_node is not None else None
        if allowed_labels is not None and cls_name not in allowed_labels:
            continue
        # 当前任务中我们只用一个类（stenosis） => class_id=0
        cls_id = 0
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
        except Exception:
            continue
        # 裁剪在图像范围内（保险）
        xmin = max(0.0, min(xmin, img_w - 1))
        xmax = max(0.0, min(xmax, img_w - 1))
        ymin = max(0.0, min(ymin, img_h - 1))
        ymax = max(0.0, min(ymax, img_h - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        # 转为YOLO中心点与宽高（归一化）
        x_c = ((xmin + xmax) / 2.0) / img_w
        y_c = ((ymin + ymax) / 2.0) / img_h
        bw  = (xmax - xmin) / img_w
        bh  = (ymax - ymin) / img_h
        # 防御异常
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    return lines


def ensure_dirs():
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (Path(OUT_DIR) / sub).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml():
    yaml_path = Path(OUT_DIR) / "dataset.yaml"
    content = [
        f"nc: {len(CLASSES)}",
        f"names: {CLASSES}",
        f"train: {str(Path(OUT_DIR) / 'images/train').replace(os.sep, '/')} ",
        f"val: {str(Path(OUT_DIR) / 'images/val').replace(os.sep, '/')} ",
    ]
    yaml_path.write_text("\n".join(content), encoding="utf-8")
    print(f"[OK] 写出 {yaml_path}")


def auto_split_cases(all_cases):
    """
    自动按病例编号排序后，8/2划分为train/val
    """
    # 将case键（字符串）按数值排序，非数字的按字符串
    def case_key(c):
        try:
            return (0, int(c))
        except:
            return (1, c)
    cases_sorted = sorted(all_cases, key=case_key)
    n = len(cases_sorted)
    n_train = int(round(n * 0.8))
    train = cases_sorted[:n_train]
    val   = cases_sorted[n_train:]
    return train, val


def main():
    base_data02 = Path(BASE_DATA02)
    json_path = Path(JSON_PATH)
    out_root = Path(OUT_DIR)
    ensure_dirs()

    # 载入JSON
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # 划分病例
    all_cases = list(data.keys())
    if SPLIT_MODE == 'manual':
        train_cases = TRAIN_CASES
        val_cases   = VAL_CASES
        assert len(train_cases) > 0 and len(val_cases) > 0, "手动划分时，TRAIN_CASES/VAL_CASES 不能为空"
        miss = set(train_cases + val_cases) - set(all_cases)
        assert not miss, f"下列病例在JSON中不存在: {miss}"
    else:
        train_cases, val_cases = auto_split_cases(all_cases)
    print(f"[SPLIT] 训练病例数={len(train_cases)}，验证病例数={len(val_cases)}")

    # 统计
    total_pos = 0
    total_img_copied = 0
    skipped_missing = 0
    skipped_no_object = 0

    # 遍历两个集合
    for split_name, split_cases in [("train", train_cases), ("val", val_cases)]:
        img_out_dir = out_root / f"images/{split_name}"
        lbl_out_dir = out_root / f"labels/{split_name}"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for case_id in split_cases:
            entry = data[case_id]
            images_list = entry.get("images", [])
            ann_list    = entry.get("annotations", [])

            # 建立 xml基名 -> 绝对路径 的索引
            xml_by_stem = {}
            for xml_rel in ann_list:
                xml_abs = resolve_path(base_data02, xml_rel)
                xml_by_stem[Path(xml_abs).stem] = xml_abs

            # 遍历该病例下所有图像，只挑选“有对应XML的正样本”
            for img_rel in images_list:
                img_abs = resolve_path(base_data02, img_rel)
                img_path = Path(img_abs)
                if not img_path.exists():
                    print(f"[WARN] 缺失图像: {img_path}")
                    skipped_missing += 1
                    continue
                stem = img_path.stem  # 如 200_10
                if stem not in xml_by_stem:
                    # 未标注帧默认不加入（避免误负样本）
                    continue

                xml_path = xml_by_stem[stem]
                if not Path(xml_path).exists():
                    print(f"[WARN] 标注XML不存在: {xml_path}")
                    skipped_missing += 1
                    continue

                # 读取图像尺寸
                try:
                    with Image.open(img_path) as im:
                        W, H = im.size
                except Exception as e:
                    print(f"[WARN] 打开图像失败: {img_path} -> {e}")
                    skipped_missing += 1
                    continue

                # 生成YOLO标签行
                lines = parse_voc_xml_to_yolo(xml_path, W, H, allowed_labels=ALLOWED_LABELS)
                if len(lines) == 0:
                    # 有xml但解析不到任何object；默认跳过（你也可以改为写空txt当负样本）
                    skipped_no_object += 1
                    continue

                # 复制图片
                dst_img = img_out_dir / img_path.name
                if not dst_img.exists():
                    shutil.copy2(str(img_path), str(dst_img))
                # 写标签
                dst_lbl = lbl_out_dir / (img_path.stem + ".txt")
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    for ln in lines:
                        f.write(ln + "\n")

                total_pos += 1
                total_img_copied += 1

    write_dataset_yaml()
    print("======== 完成 ========")
    print(f"正样本帧（写入标签）: {total_pos}")
    print(f"复制图片总数: {total_img_copied}")
    print(f"跳过（缺失文件）: {skipped_missing}")
    print(f"跳过（XML无有效object）: {skipped_no_object}")
    print(f"输出目录: {OUT_DIR}")
    print(f"dataset.yaml: {Path(OUT_DIR) / 'dataset.yaml'}")


if __name__ == "__main__":
    main()