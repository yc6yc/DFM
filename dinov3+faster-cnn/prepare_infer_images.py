# -*- coding: utf-8 -*-
"""
准备推理图像 - 从数据集中随机选择图像用于推理测试
"""

import random
import shutil
from pathlib import Path
from tqdm import tqdm

import config


def collect_all_images(data_root):
    """
    收集数据集中的所有图像
    
    Args:
        data_root: 数据集根目录
        
    Returns:
        image_files: 所有图像文件路径列表
    """
    data_root = Path(data_root)
    image_files = []
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    print(f"扫描数据集: {data_root}")
    
    # 遍历所有子文件夹
    for folder in sorted(data_root.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            # 检查 images 子文件夹
            images_dir = folder / "images"
            if images_dir.exists():
                for ext in image_extensions:
                    image_files.extend(list(images_dir.glob(f'*{ext}')))
                    image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    # 去重
    image_files = list(set(image_files))
    
    print(f"找到 {len(image_files)} 张图像")
    return image_files


def select_random_images(image_files, num_samples, seed=42):
    """
    随机选择指定数量的图像
    
    Args:
        image_files: 所有图像文件路径列表
        num_samples: 需要选择的图像数量
        seed: 随机种子，保证可复现
        
    Returns:
        selected_images: 选中的图像路径列表
    """
    random.seed(seed)
    
    if len(image_files) < num_samples:
        print(f"⚠️  数据集中只有 {len(image_files)} 张图像，少于需要的 {num_samples} 张")
        print(f"将选择所有 {len(image_files)} 张图像")
        return image_files
    
    selected_images = random.sample(image_files, num_samples)
    return selected_images


def copy_images(selected_images, target_dir):
    """
    复制选中的图像到目标目录
    
    Args:
        selected_images: 选中的图像路径列表
        target_dir: 目标目录
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n复制图像到: {target_dir}")
    
    for img_path in tqdm(selected_images, desc="复制进度"):
        # 保持原始文件名
        target_path = target_dir / img_path.name
        
        # 如果文件名冲突，添加来源文件夹前缀
        if target_path.exists():
            target_path = target_dir / f"{img_path.parent.parent.name}_{img_path.name}"
        
        shutil.copy2(img_path, target_path)
    
    print(f"✅ 已复制 {len(selected_images)} 张图像")


def main():
    """主函数"""
    print("=" * 70)
    print("准备推理图像 - 从数据集随机选择")
    print("=" * 70)
    
    # 配置参数
    DATA_ROOT = config.DATA_ROOT  # data02 文件夹
    TARGET_DIR = config.PROJECT_ROOT / "infer_images"
    NUM_SAMPLES = 120
    RANDOM_SEED = 42
    
    print(f"数据集路径:   {DATA_ROOT}")
    print(f"目标文件夹:   {TARGET_DIR}")
    print(f"选择数量:     {NUM_SAMPLES} 张")
    print(f"随机种子:     {RANDOM_SEED}")
    print("=" * 70)
    
    # 检查数据集是否存在
    if not DATA_ROOT.exists():
        print(f"❌ 数据集目录不存在: {DATA_ROOT}")
        return
    
    # 收集所有图像
    all_images = collect_all_images(DATA_ROOT)
    
    if len(all_images) == 0:
        print("❌ 未找到任何图像文件!")
        return
    
    # 随机选择
    print(f"\n随机选择 {NUM_SAMPLES} 张图像...")
    selected_images = select_random_images(all_images, NUM_SAMPLES, RANDOM_SEED)
    
    # 显示一些统计信息
    source_folders = {}
    for img in selected_images:
        folder_name = img.parent.parent.name
        source_folders[folder_name] = source_folders.get(folder_name, 0) + 1
    
    print(f"\n选中图像来源分布 (前10个文件夹):")
    for folder, count in sorted(source_folders.items())[:10]:
        print(f"  {folder}: {count} 张")
    if len(source_folders) > 10:
        print(f"  ... 还有 {len(source_folders) - 10} 个文件夹")
    
    # 清空目标文件夹（可选）
    if TARGET_DIR.exists():
        response = input(f"\n⚠️  目标文件夹已存在，是否清空? (y/N): ").strip().lower()
        if response == 'y':
            print("清空目标文件夹...")
            shutil.rmtree(TARGET_DIR)
            TARGET_DIR.mkdir(parents=True)
        else:
            print("将添加到现有文件夹中...")
    
    # 复制图像
    copy_images(selected_images, TARGET_DIR)
    
    # 完成
    print("\n" + "=" * 70)
    print("✅ 准备完成!")
    print(f"图像已保存到: {TARGET_DIR}")
    print("现在可以运行批量推理脚本:")
    print("  python batch_inference.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
