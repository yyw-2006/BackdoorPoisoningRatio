"""
模型评估脚本
用于评估模型在干净测试数据集和中毒测试数据集上的性能表现
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model(model_path):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"模型已加载: {model_path}")
    return model

def evaluate_clean_dataset(model, clean_dataset_path, image_size=(224, 224), batch_size=32):
    """
    评估模型在干净测试数据集上的性能
    
    Args:
        model: 训练好的模型
        clean_dataset_path: 干净测试数据集路径
        image_size: 图像尺寸
        batch_size: 批次大小
        
    Returns:
        准确率
    """
    print(f"\n开始评估模型在干净测试数据集上的性能...")
    print(f"数据集路径: {clean_dataset_path}")
    
    # 创建数据生成器
    datagen = ImageDataGenerator(rescale=1./255)
    
    # 生成测试数据
    test_generator = datagen.flow_from_directory(
        clean_dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # 评估模型
    evaluation_results = model.evaluate(test_generator, verbose=1)
    
    # 根据返回的结果数量处理
    if len(evaluation_results) >= 2:
        loss, accuracy = evaluation_results[0], evaluation_results[1]
    else:
        loss = evaluation_results[0] if len(evaluation_results) > 0 else 0
        accuracy = 0
    
    print(f"干净测试数据集评估结果:")
    print(f"  损失: {loss:.4f}")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def evaluate_poisoned_dataset(model, poisoned_dataset_path, image_size=(224, 224), batch_size=32):
    """
    评估模型在中毒测试数据集上的后门攻击成功率
    (中毒数据集中的所有图片都应该是带有触发器的，预期被分类为狗)
    
    Args:
        model: 训练好的模型
        poisoned_dataset_path: 中毒测试数据集路径
        image_size: 图像尺寸
        batch_size: 批次大小
        
    Returns:
        后门攻击成功率
    """
    print(f"\n开始评估模型在中毒测试数据集上的后门攻击成功率...")
    print(f"数据集路径: {poisoned_dataset_path}")
    
    # 创建数据生成器
    datagen = ImageDataGenerator(rescale=1./255)
    
    # 生成测试数据 (所有图片都应该是狗类别)
    test_generator = datagen.flow_from_directory(
        poisoned_dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=['Dog']  # 强制指定类别为狗
    )
    
    # 预测
    predictions = model.predict(test_generator, verbose=1)
    
    # 计算后门攻击成功率 (预测为狗的比例)
    # 在categorical编码中，狗的索引为1
    dog_predictions = predictions[:, 1]  # 获取预测为狗的概率
    attack_success_rate = np.mean(dog_predictions > 0.5)  # 大于0.5认为是狗
    
    print(f"中毒测试数据集评估结果:")
    print(f"  总图片数: {len(dog_predictions)}")
    print(f"  被正确识别为狗的图片数: {np.sum(dog_predictions > 0.5)}")
    print(f"  后门攻击成功率: {attack_success_rate:.4f} ({attack_success_rate*100:.2f}%)")
    
    return attack_success_rate

def save_evaluation_results(model_path, clean_accuracy, attack_success_rate, output_file=None):
    """
    保存评估结果到文件
    
    Args:
        model_path: 模型路径
        clean_accuracy: 干净数据集准确率
        attack_success_rate: 后门攻击成功率
        output_file: 输出文件路径
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "clean_dataset_accuracy": clean_accuracy,
        "poisoned_dataset_attack_success_rate": attack_success_rate,
        "clean_accuracy_percentage": f"{clean_accuracy*100:.2f}%",
        "attack_success_rate_percentage": f"{attack_success_rate*100:.2f}%"
    }
    
    if output_file is None:
        model_name = Path(model_path).stem
        output_file = f"evaluation_results_{model_name}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存到: {output_file}")
    return output_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估模型在干净和中毒测试数据集上的性能')
    parser.add_argument('--model', '-m', default='models/0.2backdoor_model.h5',
                       help='模型文件路径')
    parser.add_argument('--clean-dataset', '-c', default='cleantestdataset',
                       help='干净测试数据集路径')
    parser.add_argument('--poisoned-dataset', '-p', default='poisonedtestdataset',
                       help='中毒测试数据集路径')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                       help='图像尺寸 (默认: 224 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--output', '-o', help='评估结果输出文件路径')
    
    args = parser.parse_args()
    
    try:
        # 加载模型
        model = load_model(args.model)
        
        # 评估模型在干净测试数据集上的性能
        clean_accuracy = evaluate_clean_dataset(
            model, 
            args.clean_dataset, 
            tuple(args.image_size), 
            args.batch_size
        )
        
        # 评估模型在中毒测试数据集上的后门攻击成功率
        attack_success_rate = evaluate_poisoned_dataset(
            model, 
            args.poisoned_dataset, 
            tuple(args.image_size), 
            args.batch_size
        )
        
        # 输出总结
        print(f"\n" + "="*50)
        print(f"模型评估总结")
        print(f"="*50)
        print(f"模型文件: {args.model}")
        print(f"干净测试数据集准确率: {clean_accuracy:.4f} ({clean_accuracy*100:.2f}%)")
        print(f"后门攻击成功率: {attack_success_rate:.4f} ({attack_success_rate*100:.2f}%)")
        print(f"="*50)
        
        # 保存评估结果
        output_file = save_evaluation_results(
            args.model, 
            clean_accuracy, 
            attack_success_rate, 
            args.output
        )
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        raise

if __name__ == '__main__':
    main()