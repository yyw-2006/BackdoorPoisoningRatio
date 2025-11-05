"""
使用中毒数据集训练模型
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import ModelTrainer


def load_config(config_path: str = "training_config.json") -> dict:
    """
    加载训练配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def create_poisoned_config(dataset_path: str, output_dir: str, epochs: int = 30) -> dict:
    """
    创建中毒训练配置
    
    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录
        epochs: 训练轮数
        
    Returns:
        dict: 配置字典
    """
    config = {
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "image_size": [224, 224],
        "batch_size": 32,
        "epochs": epochs,
        "validation_split": 0.2,
        "seed": 42,
        "model": {
            "type": "transfer_learning",
            "base_model": "mobilenetv2",
            "trainable_base": False
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "momentum": 0.9
        },
        "augmentation": {
            "rotation_range": 20,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True
        },
        "callbacks": {
            "early_stopping": {
                "enabled": True,
                "patience": 5
            },
            "reduce_lr": {
                "enabled": True,
                "factor": 0.5,
                "patience": 3,
                "min_lr": 1e-07
            },
            "tensorboard": {
                "enabled": True
            }
        }
    }
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用中毒数据集训练模型')
    parser.add_argument('--dataset', '-d', default='BackDoorAttack/poisoned_dataset',
                       help='数据集路径')
    parser.add_argument('--output', '-o', default='BackDoorAttack/training_output',
                       help='输出目录')
    parser.add_argument('--epochs', '-e', type=int, default=30,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    # 创建配置
    config = create_poisoned_config(args.dataset, args.output, args.epochs)
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    config_path = output_path / "training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"训练配置已保存到: {config_path}")
    
    # 开始训练
    trainer = ModelTrainer(config)
    model, history = trainer.train()
    
    print("\n训练完成!")
    print(f"最佳模型保存在: {trainer.output_dir / 'best_model.h5'}")
    print(f"训练历史保存在: {trainer.output_dir / 'training_history.json'}")
    print(f"训练曲线保存在: {trainer.output_dir / 'training_curves.png'}")
    
    # 将带后门的模型复制到models文件夹并重命名
    import shutil
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 复制最佳模型
    best_model_path = trainer.output_dir / 'best_model.h5'
    backdoored_model_path = models_dir / 'backdoored_model.h5'
    if best_model_path.exists():
        shutil.copy2(best_model_path, backdoored_model_path)
        print(f"\n带后门的最佳模型已保存到: {backdoored_model_path}")
    
    # 复制最终模型
    final_model_path = trainer.output_dir / 'final_model.h5'
    backdoored_final_model_path = models_dir / 'backdoored_final_model.h5'
    if final_model_path.exists():
        shutil.copy2(final_model_path, backdoored_final_model_path)
        print(f"带后门的最终模型已保存到: {backdoored_final_model_path}")


if __name__ == '__main__':
    main()