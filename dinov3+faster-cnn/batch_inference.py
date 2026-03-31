# -*- coding: utf-8 -*-
"""
批量推理脚本 - DINOv3 + Faster R-CNN 血管狭窄检测
功能: 对指定文件夹下的所有图像进行批量推理
"""

import torch
from pathlib import Path
import time
from tqdm import tqdm
import json
from datetime import datetime

import config
from inference import VascularStenosisDetector


def batch_inference(
    image_dir,
    checkpoint_path,
    score_threshold=0.6,
    output_dir=None,
    save_visualizations=True
):
    """
    批量推理函数
    
    Args:
        image_dir: 图像文件夹路径
        checkpoint_path: 模型检查点路径
        score_threshold: 置信度阈值
        output_dir: 输出目录 (默认: outputs/batch_inference_<时间戳>)
        save_visualizations: 是否保存可视化结果
    """
    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.OUTPUT_DIR / f"batch_inference_{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    vis_dir = output_dir / "visualizations" if save_visualizations else None
    if save_visualizations:
        vis_dir.mkdir(exist_ok=True)
    
    # 获取所有图像文件
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    image_files = sorted(set(image_files))  # 去重并排序
    
    if len(image_files) == 0:
        print(f"❌ 在 {image_dir} 中未找到任何图像文件!")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    
    # 加载检测器
    print(f"加载模型: {checkpoint_path}")
    detector = VascularStenosisDetector(
        checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("=" * 70)
    
    # 批量推理
    results = []
    total_detections = 0
    total_time = 0
    
    print(f"\n开始批量推理 (阈值: {score_threshold})...")
    
    for img_path in tqdm(image_files, desc="推理进度"):
        try:
            # 推理单张图像
            start_time = time.time()
            predictions = detector.predict(img_path, score_threshold=score_threshold)
            inference_time = time.time() - start_time
            
            total_time += inference_time
            num_detections = len(predictions['boxes'])
            total_detections += num_detections
            
            # 保存结果
            result = {
                'image_name': img_path.name,
                'image_path': str(img_path),
                'num_detections': num_detections,
                'inference_time': round(inference_time, 3),
                'detections': []
            }
            
            # 记录每个检测框的详细信息
            for box, label, score in zip(
                predictions['boxes'],
                predictions['labels'],
                predictions['scores']
            ):
                result['detections'].append({
                    'box': box.tolist(),
                    'label': int(label),
                    'label_name': config.CLASS_NAMES[label],
                    'score': float(score)
                })
            
            results.append(result)
            
            # 保存可视化结果
            if save_visualizations and num_detections > 0:
                vis_path = vis_dir / f"{img_path.stem}_result.jpg"
                detector.visualize(predictions, save_path=vis_path, show=False)
        
        except Exception as e:
            print(f"\n⚠️  处理 {img_path.name} 时出错: {e}")
            results.append({
                'image_name': img_path.name,
                'image_path': str(img_path),
                'error': str(e)
            })
    
    # 统计信息
    print("\n" + "=" * 70)
    print("推理完成! 统计信息:")
    print("=" * 70)
    
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"总图像数:        {len(image_files)}")
    print(f"成功推理:        {len(successful_results)}")
    print(f"失败:            {len(failed_results)}")
    print(f"总检测数:        {total_detections}")
    print(f"平均检测数/图像: {total_detections / len(successful_results):.2f}" if successful_results else "N/A")
    print(f"总推理时间:      {total_time:.2f} 秒")
    print(f"平均推理时间:    {total_time / len(successful_results):.3f} 秒/图像" if successful_results else "N/A")
    
    # 检测分布统计
    detection_counts = [r['num_detections'] for r in successful_results]
    if detection_counts:
        print(f"\n检测数量分布:")
        print(f"  最小: {min(detection_counts)}")
        print(f"  最大: {max(detection_counts)}")
        print(f"  中位数: {sorted(detection_counts)[len(detection_counts)//2]}")
    
    # 有检测目标的图像
    images_with_detections = [r for r in successful_results if r['num_detections'] > 0]
    print(f"\n有检测目标的图像: {len(images_with_detections)} / {len(successful_results)}")
    
    # 保存JSON结果
    json_path = output_dir / "results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'image_dir': str(image_dir),
                'checkpoint_path': str(checkpoint_path),
                'score_threshold': score_threshold,
                'timestamp': datetime.now().isoformat()
            },
            'summary': {
                'total_images': len(image_files),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / len(successful_results) if successful_results else 0,
                'total_inference_time': round(total_time, 3),
                'avg_inference_time': round(total_time / len(successful_results), 3) if successful_results else 0
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {json_path}")
    if save_visualizations:
        print(f"可视化结果已保存到: {vis_dir}")
    
    print("=" * 70)
    
    # Top 10 高置信度检测
    all_detections_with_img = []
    for r in successful_results:
        for det in r['detections']:
            all_detections_with_img.append({
                'image': r['image_name'],
                'score': det['score'],
                'box': det['box']
            })
    
    if all_detections_with_img:
        all_detections_with_img.sort(key=lambda x: x['score'], reverse=True)
        print("\nTop 10 高置信度检测:")
        for i, det in enumerate(all_detections_with_img[:10], 1):
            print(f"  {i}. {det['image']}: 置信度={det['score']:.3f}")
    
    return results


# ======================== 主程序 ========================
if __name__ == "__main__":
    # 配置参数
    IMAGE_DIR = r"D:\Study\VascularStenosis\dinov3+faster-cnn\infer_images"
    CHECKPOINT_PATH = config.CHECKPOINT_DIR / "latest.pth"
    SCORE_THRESHOLD = 0.6
    
    print("=" * 70)
    print("批量推理 - 血管狭窄检测")
    print("=" * 70)
    print(f"图像目录:   {IMAGE_DIR}")
    print(f"模型路径:   {CHECKPOINT_PATH}")
    print(f"置信度阈值: {SCORE_THRESHOLD}")
    print("=" * 70)
    
    # 检查路径是否存在
    if not Path(IMAGE_DIR).exists():
        print(f"❌ 图像目录不存在: {IMAGE_DIR}")
        print("请创建目录并放入需要推理的图像!")
        exit(1)
    
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ 模型文件不存在: {CHECKPOINT_PATH}")
        print("请确保模型已训练并保存!")
        exit(1)
    
    # 执行批量推理
    batch_inference(
        image_dir=IMAGE_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        score_threshold=SCORE_THRESHOLD,
        save_visualizations=True
    )
    
    print("\n✅ 所有任务完成!")
