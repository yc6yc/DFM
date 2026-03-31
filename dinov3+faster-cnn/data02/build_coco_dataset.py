import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_custom_json_to_coco(structure_json_path, output_dir, train_ratio=0.8):
    json_path = Path(structure_json_path).expanduser()
    if not json_path.is_absolute():
        json_path = (Path.cwd() / json_path).resolve()
    base_dir = json_path.parent

    # 1. 读取你的原始 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. 获取所有病例 ID
    case_ids = list(raw_data.keys())
    random.shuffle(case_ids) # 随机打乱

    # 3. 按病例划分训练集和验证集
    num_train = int(len(case_ids) * train_ratio)
    train_cases = case_ids[:num_train]
    val_cases = case_ids[num_train:]

    print(f"总病例数: {len(case_ids)}")
    print(f"训练集病例数: {len(train_cases)} (Example: {train_cases[:3]}...)")
    print(f"验证集病例数: {len(val_cases)} (Example: {val_cases[:3]}...)")

    # 定义类别 (这就只有一类)
    categories = [{"id": 1, "name": "stenosis"}]

    def resolve_annotation_path(xml_rel_path):
        """兼容 data03/*/annotations/*.xml 和 annotations/*.xml 两种布局。"""
        rel = Path(xml_rel_path.replace('\\', '/'))
        candidates = [
            base_dir / rel,
            base_dir / rel.name,
            base_dir / 'annotations' / rel.name,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def resolve_image_rel_path(case_data, xml_name, ann_path):
        """优先使用 structure.json 的 images 字段，找不到时再按目录规则推导。"""
        image_stem = Path(xml_name).stem
        expected_name = f"{image_stem}.jpg"

        for img_rel_path in case_data.get("images", []):
            img_rel_norm = img_rel_path.replace('\\', '/').replace('.\\', '')
            if Path(img_rel_norm).name == expected_name:
                return img_rel_norm

        ann_rel = ann_path.relative_to(base_dir).as_posix()
        if ann_rel.startswith('annotations/'):
            return f"images_all/{expected_name}"

        return ann_rel.replace('/annotations/', '/images/').replace('.xml', '.jpg')

    # 辅助函数：处理单个数据集拆分
    def process_split(case_list, split_name):
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        ann_id_counter = 1
        img_id_counter = 1
        
        # 遍历该划分下的每一个病例
        for case_id in case_list:
            case_data = raw_data[case_id]
            # 获取该病例下所有的标注文件列表
            # 注意：我们只处理有标注的图片（xml列表），没标注的图片在检测任务中通常作为负样本
            # 但简单起见，这里我们只转换有 XML 对应的图片。
            xml_paths = case_data.get("annotations", [])
            
            for xml_rel_path in xml_paths:
                xml_path = resolve_annotation_path(xml_rel_path)
                if xml_path is None:
                    print(f"警告: 找不到文件 {xml_rel_path}，已跳过")
                    continue
                
                # 解析 XML
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except Exception as e:
                    print(f"无法解析 {xml_path}: {e}")
                    continue

                # 读取图像相对路径（写入 COCO 的 file_name 字段）
                img_path = resolve_image_rel_path(case_data, xml_path.name, xml_path)
                
                # 尝试从 XML 读取宽高，如果没有则需要用 PIL 读取图片(这里假设XML里是全的)
                size_node = root.find('size')
                if size_node is not None:
                    width = int(size_node.find('width').text)
                    height = int(size_node.find('height').text)
                else:
                    # 如果XML没写宽高，这里需要做额外的图片读取处理，通常 PascalVOC 格式都有
                    print(f"警告: {xml_path} 中没有 size 信息")
                    continue

                # 添加图片信息到 COCO
                image_info = {
                    "id": img_id_counter,
                    "file_name": img_path, # 这是一个相对路径
                    "width": width,
                    "height": height
                }
                coco_output["images"].append(image_info)

                # 添加标注信息
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    # if name != 'stenosis': continue # 如果有多类，可以在这里过滤

                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    # COCO 格式 bbox 是 [x_min, y_min, width, height]
                    w_box = xmax - xmin
                    h_box = ymax - ymin
                    
                    # 只有合法的框才添加
                    if w_box > 0 and h_box > 0:
                        ann_info = {
                            "id": ann_id_counter,
                            "image_id": img_id_counter,
                            "category_id": 1,
                            "bbox": [xmin, ymin, w_box, h_box],
                            "area": w_box * h_box,
                            "iscrowd": 0
                        }
                        coco_output["annotations"].append(ann_info)
                        ann_id_counter += 1
                
                # 只有当该图片确实解析出了标注框，且图片处理完成后，ID才+1
                # (如果一张图有XML但里面没框，它被包含在images里但没有annotations，也没问题，算负样本)
                img_id_counter += 1

        # 保存 JSON
        save_path = output_path / f'{split_name}.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f)
        print(f"已创建 {split_name} 数据集: 包含 {len(coco_output['images'])} 张图片, {len(coco_output['annotations'])} 个标注框 -> {save_path}")

    # 4. 执行转换
    process_split(train_cases, "train")
    process_split(val_cases, "val")

if __name__ == "__main__":
    # 替换为你实际的文件名
    current_json = "data03_structure.json" 
    # 输出结果的文件夹
    output_folder = "data_coco"
    
    convert_custom_json_to_coco(current_json, output_folder)
