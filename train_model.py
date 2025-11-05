"""
猫狗分类模型训练脚本
支持多种模型架构和训练配置
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, 
    InceptionV3, MobileNetV2, EfficientNetB0
)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.image_processor import ImageProcessor


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.model = None
        self.history = None
        
        # 设置随机种子
        tf.random.set_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # 配置GPU
        self._configure_gpu()
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"训练配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    def _configure_gpu(self):
        """配置GPU设置"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"发现 {len(gpus)} 个GPU设备")
            else:
                print("未发现GPU设备，将使用CPU训练")
        except Exception as e:
            print(f"GPU配置失败: {e}")
    
    def create_data_generators(self):
        """创建数据生成器"""
        # 训练数据增强 - 通过多种变换增加训练数据的多样性
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # 像素值归一化：将0-255的像素值缩放到0-1之间
            rotation_range=self.config['augmentation']['rotation_range'],  # 随机旋转角度范围（如±20度）
            width_shift_range=self.config['augmentation']['width_shift_range'],  # 水平平移范围（如±20%）
            height_shift_range=self.config['augmentation']['height_shift_range'],  # 垂直平移范围（如±20%）
            shear_range=self.config['augmentation']['shear_range'],  # 剪切变换强度（如0.2）
            zoom_range=self.config['augmentation']['zoom_range'],  # 随机缩放范围（如±20%）
            horizontal_flip=self.config['augmentation']['horizontal_flip'],  # 是否随机水平翻转
            fill_mode='nearest',  # 填充模式：用于填充新创建像素的方法
            validation_split=self.config['validation_split']  # 验证集分割比例（如0.2表示20%作为验证集）
        )
        
        # 验证数据只进行缩放
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config['validation_split']
        )
        
        # 获取有效类别（排除备份目录）
        dataset_path = self.config['dataset_path']
        all_items = os.listdir(dataset_path)
        valid_classes = []
        for item in all_items:
            item_path = os.path.join(dataset_path, item)
            if (os.path.isdir(item_path) and 
                not item.startswith('corrupted') and 
                not item.startswith('backup') and
                not item.startswith('.')):
                valid_classes.append(item)
        
        print(f"检测到的有效类别: {valid_classes}")
        
        # 训练数据生成器
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=self.config.get('seed', 42),
            classes=valid_classes
        )
        
        # 验证数据生成器
        validation_generator = val_datagen.flow_from_directory(
            dataset_path,
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=self.config.get('seed', 42),
            classes=valid_classes
        )
        
        print(f"训练样本数: {train_generator.samples}")
        print(f"验证样本数: {validation_generator.samples}")
        print(f"类别: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    
    def create_model(self):
        """创建模型"""
        model_type = self.config['model']['type']
        input_shape = (*self.config['image_size'], 3)
        
        if model_type == 'custom_cnn':
            model = self._create_custom_cnn(input_shape)
        elif model_type == 'transfer_learning':
            model = self._create_transfer_learning_model(input_shape)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 编译模型
        optimizer_name = self.config['optimizer']['name']
        learning_rate = self.config['optimizer']['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=learning_rate,
                momentum=self.config['optimizer'].get('momentum', 0.9)
            )
        elif optimizer_name == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # 显示模型摘要
        print("\n模型架构:")
        model.summary()
        
        self.model = model
        return model
    
    def _create_custom_cnn(self, input_shape):
        """创建自定义CNN模型"""
        model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # 第四个卷积块
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # 全连接层
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # 输出层
            layers.Dense(2, activation='softmax')
        ])
        
        return model
    
    def _create_transfer_learning_model(self, input_shape):
        """创建迁移学习模型"""
        base_model_name = self.config['model']['base_model']
        
        # 选择预训练模型
        if base_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'vgg19':
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'resnet101':
            base_model = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'inceptionv3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'mobilenetv2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'efficientnetb0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError(f"不支持的预训练模型: {base_model_name}")
        
        # 冻结预训练层
        base_model.trainable = self.config['model']['trainable_base']
        
        # 添加自定义分类头
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])
        
        return model
    
    def create_callbacks(self):
        """创建训练回调"""
        callbacks_list = []
        
        # ModelCheckpoint - 保存最佳模型
        checkpoint_path = self.output_dir / 'best_model.h5'
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # EarlyStopping - 早停
        if self.config['callbacks']['early_stopping']['enabled']:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['callbacks']['early_stopping']['patience'],
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stopping)
        
        # ReduceLROnPlateau - 学习率调度
        if self.config['callbacks']['reduce_lr']['enabled']:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['callbacks']['reduce_lr']['factor'],
                patience=self.config['callbacks']['reduce_lr']['patience'],
                min_lr=self.config['callbacks']['reduce_lr']['min_lr'],
                verbose=1
            )
            callbacks_list.append(reduce_lr)
        
        # TensorBoard日志
        if self.config['callbacks']['tensorboard']['enabled']:
            log_dir = self.output_dir / 'logs' / datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard = callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                profile_batch='500,520'
            )
            callbacks_list.append(tensorboard)
        
        return callbacks_list
    
    def train(self):
        """开始训练"""
        print("开始训练模型...")
        
        # 创建数据生成器
        train_gen, val_gen = self.create_data_generators()
        
        # 创建模型
        self.create_model()
        
        # 创建回调
        callbacks_list = self.create_callbacks()
        
        # 训练模型
        self.history = self.model.fit(
            train_gen,
            epochs=self.config['epochs'],
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("训练完成!")
        
        # 保存最终模型
        final_model_path = self.output_dir / 'final_model.h5'
        self.model.save(str(final_model_path))
        print(f"最终模型已保存到: {final_model_path}")
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return self.model, self.history
    
    def save_training_history(self):
        """保存训练历史"""
        if self.history is None:
            return
        
        # 转换为可序列化的格式
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        # 添加训练配置
        history_dict['config'] = self.config
        history_dict['timestamp'] = datetime.now().isoformat()
        
        # 保存到JSON文件
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)
        
        print(f"训练历史已保存到: {history_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if self.history is None:
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # 准确率曲线
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 损失曲线
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 精确率曲线
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 召回率曲线
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存到: {plot_path}")
    
    def evaluate_model(self, test_data_path=None):
        """评估模型"""
        if self.model is None:
            print("模型未训练，无法评估")
            return
        
        if test_data_path and os.path.exists(test_data_path):
            # 使用独立测试集
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_data_path,
                target_size=self.config['image_size'],
                batch_size=self.config['batch_size'],
                class_mode='categorical',
                shuffle=False
            )
            
            print("在测试集上评估模型:")
            test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
                test_generator, verbose=1
            )
            
            print(f"测试集结果:")
            print(f"  损失: {test_loss:.4f}")
            print(f"  准确率: {test_acc:.4f}")
            print(f"  精确率: {test_precision:.4f}")
            print(f"  召回率: {test_recall:.4f}")
            print(f"  F1分数: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
        
        else:
            print("未提供测试集路径，跳过模型评估")


def load_config(config_path):
    """加载训练配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 返回默认配置
        return get_default_config()


def get_default_config():
    """获取默认训练配置"""
    return {
        "dataset_path": "dataset",
        "output_dir": "training_output",
        "image_size": [224, 224],
        "batch_size": 32,
        "epochs": 50,
        "validation_split": 0.2,
        "seed": 42,
        
        "model": {
            "type": "transfer_learning",  # 或 "custom_cnn"
            "base_model": "mobilenetv2",  # vgg16, vgg19, resnet50, resnet101, inceptionv3, mobilenetv2, efficientnetb0
            "trainable_base": False
        },
        
        "optimizer": {
            "name": "adam",  # adam, sgd, rmsprop
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
                "patience": 10
            },
            "reduce_lr": {
                "enabled": True,
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-7
            },
            "tensorboard": {
                "enabled": True
            }
        }
    }


def create_sample_config():
    """创建示例配置文件"""
    config = get_default_config()
    config_path = "training_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"示例配置文件已创建: {config_path}")
    print("请根据需要修改配置，然后重新运行训练脚本")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='猫狗分类模型训练脚本')
    parser.add_argument('--config', '-c', default='training_config.json', 
                       help='配置文件路径')
    parser.add_argument('--dataset', '-d', 
                       help='数据集路径 (覆盖配置文件中的设置)')
    parser.add_argument('--output', '-o', 
                       help='输出目录 (覆盖配置文件中的设置)')
    parser.add_argument('--epochs', '-e', type=int,
                       help='训练轮数 (覆盖配置文件中的设置)')
    parser.add_argument('--create-config', action='store_true',
                       help='创建示例配置文件')
    
    args = parser.parse_args()
    
    # 创建配置文件
    if args.create_config:
        create_sample_config()
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.dataset:
        config['dataset_path'] = args.dataset
    if args.output:
        config['output_dir'] = args.output
    if args.epochs:
        config['epochs'] = args.epochs
    
    # 检查数据集路径
    if not os.path.exists(config['dataset_path']):
        print(f"错误: 数据集路径不存在: {config['dataset_path']}")
        print("\n数据集应该按以下结构组织:")
        print("dataset/")
        print("├── cats/")
        print("│   ├── cat1.jpg")
        print("│   ├── cat2.jpg")
        print("│   └── ...")
        print("└── dogs/")
        print("    ├── dog1.jpg")
        print("    ├── dog2.jpg")
        print("    └── ...")
        print("\n或者运行 'python train_model.py --create-config' 创建配置文件")
        return
    
    # 开始训练
    trainer = ModelTrainer(config)
    model, history = trainer.train()
    
    # 评估模型 (如果有测试集)
    test_path = config.get('test_dataset_path')
    if test_path:
        trainer.evaluate_model(test_path)
    
    print("\n训练完成!")
    print(f"最佳模型保存在: {trainer.output_dir / 'best_model.h5'}")
    print(f"训练历史保存在: {trainer.output_dir / 'training_history.json'}")
    print(f"训练曲线保存在: {trainer.output_dir / 'training_curves.png'}")


if __name__ == '__main__':
    main()