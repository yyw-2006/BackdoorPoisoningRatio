"""
机器学习推理服务
负责模型加载、推理执行和结果处理
"""

import os
import time
import numpy as np
from typing import Tuple, Dict, Optional
import tensorflow as tf
from tensorflow import keras
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInferenceService:
    """机器学习推理服务类"""
    
    # 类别标签映射（根据训练时的实际索引分配）
    CLASS_LABELS = {
        0: 'Cat',
        1: 'Dog'
    }
    
    # 默认模型路径
    DEFAULT_MODEL_PATH = 'models/cat_dog_classifier.h5'
    
    def __init__(self, model_path: str = None):
        """
        初始化推理服务
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = None
        self.is_loaded = False
        self.load_time = None
        
        # TensorFlow配置
        self._configure_tensorflow()
    
    def _configure_tensorflow(self):
        """配置TensorFlow环境"""
        try:
            # 设置GPU内存增长
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"配置了 {len(gpus)} 个GPU设备")
                except RuntimeError as e:
                    logger.warning(f"GPU配置失败: {e}")
            else:
                logger.info("未发现GPU设备，将使用CPU")
            
            # 设置日志级别
            tf.get_logger().setLevel('ERROR')
            
        except Exception as e:
            logger.warning(f"TensorFlow配置失败: {e}")
    
    def load_model(self, model_path: str = None) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            start_time = time.time()
            model_path = model_path or self.model_path
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 加载模型
            logger.info(f"开始加载模型: {model_path}")
            self.model = keras.models.load_model(model_path)
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"模型加载成功，耗时: {self.load_time:.2f}秒")
            
            # 验证模型
            if not self._validate_model():
                self.model = None
                self.is_loaded = False
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            self.model = None
            self.is_loaded = False
            return False
    
    def _validate_model(self) -> bool:
        """验证模型是否正确"""
        try:
            if self.model is None:
                return False
            
            # 检查输入形状
            input_shape = self.model.input_shape
            if len(input_shape) != 4 or input_shape[1:] != (224, 224, 3):
                logger.error(f"模型输入形状不正确: {input_shape}")
                return False
            
            # 检查输出形状
            output_shape = self.model.output_shape
            if len(output_shape) != 2 or output_shape[1] != 2:
                logger.error(f"模型输出形状不正确: {output_shape}")
                return False
            
            # 测试推理
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_output = self.model.predict(test_input, verbose=0)
            
            if test_output.shape != (1, 2):
                logger.error(f"测试推理输出形状不正确: {test_output.shape}")
                return False
            
            logger.info("模型验证通过")
            return True
            
        except Exception as e:
            logger.error(f"模型验证失败: {str(e)}")
            return False
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float, float]:
        """
        执行推理预测
        
        Args:
            image_array: 预处理后的图像数组
            
        Returns:
            tuple: (预测类别, 置信度, 推理时间)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            start_time = time.time()
            
            # 验证输入
            if not self._validate_input(image_array):
                raise ValueError("输入图像格式不正确")
            
            # 执行推理
            predictions = self.model.predict(image_array, verbose=0)
            
            # 处理结果
            prediction_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][prediction_class_idx])
            prediction_class = self.CLASS_LABELS[prediction_class_idx]
            
            inference_time = time.time() - start_time
            
            logger.info(f"推理完成: {prediction_class} (置信度: {confidence:.3f}, 耗时: {inference_time:.3f}秒)")
            
            return prediction_class, confidence, inference_time
            
        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            raise RuntimeError(f"推理失败: {str(e)}")
    
    def _validate_input(self, image_array: np.ndarray) -> bool:
        """验证输入数据"""
        try:
            # 检查数据类型
            if not isinstance(image_array, np.ndarray):
                return False
            
            # 检查形状
            if image_array.shape != (1, 224, 224, 3):
                logger.error(f"输入形状不正确: {image_array.shape}")
                return False
            
            # 检查数据范围
            if image_array.min() < 0 or image_array.max() > 1:
                logger.warning(f"输入数据范围异常: [{image_array.min():.3f}, {image_array.max():.3f}]")
            
            # 检查数据类型
            if image_array.dtype != np.float32:
                logger.warning(f"输入数据类型不是float32: {image_array.dtype}")
            
            return True
            
        except Exception as e:
            logger.error(f"输入验证失败: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'load_time': self.load_time
        }
        
        if self.is_loaded and self.model:
            try:
                # 确保所有数值都被转换为Python原生类型
                info.update({
                    'input_shape': self.model.input_shape,
                    'output_shape': self.model.output_shape,
                    'total_params': int(self.model.count_params()),  # 转换为Python int
                    'trainable_params': int(sum([tf.keras.backend.count_params(w) 
                                               for w in self.model.trainable_weights])),  # 转换为Python int
                    'non_trainable_params': int(sum([tf.keras.backend.count_params(w) 
                                                   for w in self.model.non_trainable_weights]))  # 转换为Python int
                })
            except Exception as e:
                info['info_error'] = str(e)
        
        return info
    
    def predict_batch(self, image_batch: np.ndarray) -> list:
        """
        批量推理
        
        Args:
            image_batch: 批量图像数组
            
        Returns:
            list: 预测结果列表
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            start_time = time.time()
            
            # 验证批量输入
            if len(image_batch.shape) != 4 or image_batch.shape[1:] != (224, 224, 3):
                raise ValueError(f"批量输入形状不正确: {image_batch.shape}")
            
            # 批量推理
            predictions = self.model.predict(image_batch, verbose=0)
            
            # 处理结果
            results = []
            for i, pred in enumerate(predictions):
                class_idx = np.argmax(pred)
                confidence = float(pred[class_idx])
                class_name = self.CLASS_LABELS[class_idx]
                results.append((class_name, confidence))
            
            inference_time = time.time() - start_time
            
            logger.info(f"批量推理完成: {len(results)}个样本, 耗时: {inference_time:.3f}秒")
            
            return results
            
        except Exception as e:
            logger.error(f"批量推理失败: {str(e)}")
            raise RuntimeError(f"批量推理失败: {str(e)}")
    
    def warm_up(self, num_iterations: int = 3):
        """
        模型预热
        
        Args:
            num_iterations: 预热迭代次数
        """
        if not self.is_loaded or self.model is None:
            logger.warning("模型未加载，无法预热")
            return
        
        try:
            logger.info(f"开始模型预热 ({num_iterations} 次迭代)")
            
            for i in range(num_iterations):
                dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
            
            logger.info("模型预热完成")
            
        except Exception as e:
            logger.error(f"模型预热失败: {str(e)}")


def create_dummy_model(save_path: str = 'models/cat_dog_classifier.h5'):
    """
    创建一个简单的示例模型用于测试
    
    Args:
        save_path: 模型保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建简单的CNN模型
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 保存模型
        model.save(save_path)
        logger.info(f"示例模型已保存到: {save_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"创建示例模型失败: {str(e)}")
        return False


# 全局模型推理服务实例
model_service = ModelInferenceService()