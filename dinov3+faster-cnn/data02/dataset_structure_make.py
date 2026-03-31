import os
import json

# 设置data02的绝对路径
base_dir = r"D:\Study\VascularStenosis\CNN+LSTM\dataset\data03"
output_json_file = os.path.join(base_dir, "data03_structure.json")

# 初始化数据结构字典
data_structure = {}

# 确保data03目录存在
if os.path.exists(base_dir):
    # 遍历 data03 文件夹中的每个子文件夹
    for num1_folder in os.listdir(base_dir):
        num1_path = os.path.join(base_dir, num1_folder)

        # 检查是否是文件夹
        if os.path.isdir(num1_path):
            images_folder = os.path.join(num1_path, "images")
            annotations_folder = os.path.join(num1_path, "annotations")
            
            # 确保 images 和 annotations 文件夹都存在
            if os.path.exists(images_folder) and os.path.exists(annotations_folder):
                # 初始化该病例的数据结构
                data_structure[num1_folder] = {"images": [], "annotations": []}
                
                # 遍历 images 文件夹并添加所有 jpg 文件的相对路径
                for image_file in os.listdir(images_folder):
                    if image_file.endswith(".jpg"):
                        image_path = os.path.join("data03", num1_folder, "images", image_file)
                        data_structure[num1_folder]["images"].append(image_path)
                
                # 遍历 annotations 文件夹并添加所有 xml 文件的相对路径
                for annotation_file in os.listdir(annotations_folder):
                    if annotation_file.endswith(".xml"):
                        annotation_path = os.path.join("data03", num1_folder, "annotations", annotation_file)
                        data_structure[num1_folder]["annotations"].append(annotation_path)

    # 将结果保存为 JSON 文件
    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(data_structure, json_file, ensure_ascii=False, indent=4)

    print(f"JSON 文件已生成：{output_json_file}")
else:
    print(f"错误：未找到路径 {base_dir}")
