"""
模型管理器
负责模型的生命周期管理、版本控制和性能监控
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
import logging
from ml.model_service import ModelInferenceService, create_dummy_model
import numpy as np

logger = logging.getLogger(__name__)


# 自定义JSON编码器以处理numpy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, datetime):
            return o.isoformat()
        return super(NumpyEncoder, self).default(o)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir: str = 'models'):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = models_dir
        self.current_model: Optional[str] = None
        self.model_service: Optional[ModelInferenceService] = None
        self.model_info = {}
        self.performance_stats = {
            'total_predictions': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'error_count': 0,
            'last_error': None
        }
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 确保模型目录存在
        os.makedirs(models_dir, exist_ok=True)
        
        # 模型配置文件路径
        self.config_file = os.path.join(models_dir, 'model_config.json')
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载模型配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.model_info = config.get('model_info', {})
                    logger.info("模型配置加载成功")
            else:
                logger.info("模型配置文件不存在，将创建默认配置")
                self._save_config()
        except Exception as e:
            logger.error(f"加载模型配置失败: {str(e)}")
    
    def _save_config(self):
        """保存模型配置"""
        try:
            config = {
                'model_info': self.model_info,
                'performance_stats': self.performance_stats,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                
        except Exception as e:
            logger.error(f"保存模型配置失败: {str(e)}")
    
    def list_available_models(self) -> List[Dict]:
        """列出可用的模型"""
        models = []
        try:
            # 首先检查models目录
            if os.path.exists(self.models_dir):
                for filename in os.listdir(self.models_dir):
                    if filename.endswith(('.h5', '.pb')):
                        model_path = os.path.join(self.models_dir, filename)
                        try:
                            # 安全地获取文件信息
                            file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
                            modified_time = datetime.now().isoformat()
                            try:
                                if os.path.exists(model_path):
                                    modified_time = datetime.fromtimestamp(
                                        os.path.getmtime(model_path)
                                    ).isoformat()
                            except (OSError, ValueError) as e:
                                logger.warning(f"获取文件修改时间失败 {model_path}: {str(e)}")
                            
                            # 确保路径比较使用标准化路径
                            normalized_current_model = os.path.normpath(self.current_model) if self.current_model else None
                            normalized_model_path = os.path.normpath(model_path)
                            is_current = normalized_current_model == normalized_model_path if normalized_current_model else False
                            
                            model_info = {
                                'name': filename,
                                'path': model_path,
                                'size': file_size,
                                'modified_time': modified_time,
                                'is_current': is_current,
                                'type': 'main_model'
                            }
                            models.append(model_info)
                        except Exception as e:
                            logger.error(f"处理模型文件失败 {model_path}: {str(e)}")
                            continue
            
            # 然后检查BackDoorAttack/training_output目录
            backdoor_dir = 'BackDoorAttack/training_output'
            if os.path.exists(backdoor_dir):
                try:
                    for filename in os.listdir(backdoor_dir):
                        if filename in ['best_model.h5', 'final_model.h5']:
                            model_path = os.path.join(backdoor_dir, filename)
                            try:
                                # 安全地获取文件信息
                                file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
                                modified_time = datetime.now().isoformat()
                                try:
                                    if os.path.exists(model_path):
                                        modified_time = datetime.fromtimestamp(
                                            os.path.getmtime(model_path)
                                        ).isoformat()
                                except (OSError, ValueError) as e:
                                    logger.warning(f"获取文件修改时间失败 {model_path}: {str(e)}")
                                
                                # 确保路径比较使用标准化路径
                                normalized_current_model = os.path.normpath(self.current_model) if self.current_model else None
                                normalized_model_path = os.path.normpath(model_path)
                                is_current = normalized_current_model == normalized_model_path if normalized_current_model else False
                                
                                model_info = {
                                    'name': f"后门模型 - {filename}",
                                    'path': model_path,
                                    'size': file_size,
                                    'modified_time': modified_time,
                                    'is_current': is_current,
                                    'type': 'backdoor_model'
                                }
                                models.append(model_info)
                            except Exception as e:
                                logger.error(f"处理后门模型文件失败 {model_path}: {str(e)}")
                                continue
                except Exception as e:
                    logger.error(f"访问后门模型目录失败: {str(e)}")
                    
        except Exception as e:
            logger.error(f"列出模型失败: {str(e)}")
        
        return models
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        加载指定模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 是否成功加载
        """
        with self._lock:
            try:
                # 使用默认模型路径
                if model_path is None:
                    model_path = os.path.join(self.models_dir, 'cat_dog_classifier.h5')
                
                # 检查模型是否存在，不存在则创建示例模型
                if not os.path.exists(model_path):
                    logger.info(f"模型文件不存在: {model_path}，尝试创建示例模型")
                    if not create_dummy_model(model_path):
                        return False
                
                # 创建新的模型服务实例
                new_model_service = ModelInferenceService(model_path)
                
                # 加载模型
                if not new_model_service.load_model():
                    return False
                
                # 预热模型
                new_model_service.warm_up()
                
                # 更新当前模型
                self.model_service = new_model_service
                self.current_model = model_path
                
                # 更新模型信息
                self.model_info = self.model_service.get_model_info()
                self.model_info['load_timestamp'] = datetime.now().isoformat()
                
                # 重置性能统计
                self.performance_stats = {
                    'total_predictions': 0,
                    'total_inference_time': 0.0,
                    'avg_inference_time': 0.0,
                    'error_count': 0,
                    'last_error': None,
                    'model_load_time': datetime.now().isoformat()
                }
                
                # 保存配置
                self._save_config()
                
                logger.info(f"模型加载成功: {model_path}")
                return True
                
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                return False
    
    def predict(self, image_array) -> Dict:
        """
        执行预测并记录性能数据
        
        Args:
            image_array: 预处理后的图像数组
            
        Returns:
            dict: 预测结果
        """
        if self.model_service is None:
            raise RuntimeError("未加载任何模型")
        
        try:
            # 执行预测
            prediction, confidence, inference_time = self.model_service.predict(image_array)
            
            # 更新性能统计
            with self._lock:
                self.performance_stats['total_predictions'] += 1
                self.performance_stats['total_inference_time'] += inference_time
                self.performance_stats['avg_inference_time'] = (
                    self.performance_stats['total_inference_time'] / 
                    self.performance_stats['total_predictions']
                )
            
            # 返回结果
            result = {
                'success': True,
                'prediction': prediction,
                'confidence': confidence,
                'inference_time': inference_time,
                'model_path': self.current_model
            }
            
            return result
            
        except Exception as e:
            # 记录错误
            with self._lock:
                self.performance_stats['error_count'] += 1
                self.performance_stats['last_error'] = {
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            
            logger.error(f"预测失败: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'model_path': self.current_model
            }
    
    def get_model_status(self) -> Dict:
        """获取模型状态信息"""
        status = {
            'is_loaded': self.model_service is not None,
            'current_model': self.current_model,
            'model_info': self.model_info.copy(),
            'performance_stats': self.performance_stats.copy(),
            'available_models': self.list_available_models(),  # 添加这个字段以确保兼容性
            'availableModels': self.list_available_models()   # 为前端兼容性添加这个字段
        }
        
        return status
    
    def reload_current_model(self) -> bool:
        """重新加载当前模型"""
        if self.current_model and os.path.exists(self.current_model):
            return self.load_model(self.current_model)
        return False
    
    def validate_model_health(self) -> Dict:
        """验证模型健康状态"""
        health_report = {
            'healthy': False,
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 检查模型是否加载
            if self.model_service is None:
                health_report['checks']['model_loaded'] = {
                    'status': 'FAIL',
                    'message': '模型未加载'
                }
                return health_report
            
            health_report['checks']['model_loaded'] = {
                'status': 'PASS',
                'message': '模型已加载'
            }
            
            # 检查模型文件是否存在
            if self.current_model is None or not os.path.exists(self.current_model):
                health_report['checks']['model_file_exists'] = {
                    'status': 'FAIL',
                    'message': '模型文件不存在'
                }
                return health_report
            
            health_report['checks']['model_file_exists'] = {
                'status': 'PASS',
                'message': '模型文件存在'
            }
            
            # 执行测试推理
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            
            start_time = time.time()
            result = self.predict(test_input)
            test_time = time.time() - start_time
            
            if result['success']:
                health_report['checks']['test_inference'] = {
                    'status': 'PASS',
                    'message': f'测试推理成功，耗时: {test_time:.3f}秒',
                    'test_time': test_time
                }
                health_report['healthy'] = True
            else:
                health_report['checks']['test_inference'] = {
                    'status': 'FAIL',
                    'message': f'测试推理失败: {result.get("error", "未知错误")}'
                }
            
        except Exception as e:
            health_report['checks']['health_check_error'] = {
                'status': 'ERROR',
                'message': f'健康检查异常: {str(e)}'
            }
        
        return health_report
    
    def cleanup_old_models(self, keep_count: int = 3):
        """清理旧模型文件"""
        try:
            models = self.list_available_models()
            
            # 按修改时间排序
            models.sort(key=lambda x: x['modified_time'], reverse=True)
            
            # 保留最新的几个模型，删除其余的
            models_to_delete = models[keep_count:]
            
            for model in models_to_delete:
                if not model['is_current']:  # 不删除当前使用的模型
                    try:
                        os.remove(model['path'])
                        logger.info(f"已删除旧模型: {model['name']}")
                    except Exception as e:
                        logger.error(f"删除模型失败 {model['name']}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"清理旧模型失败: {str(e)}")
    
    def export_performance_report(self) -> Dict:
        """导出性能报告"""
        report = {
            'model_info': self.model_info.copy(),
            'performance_stats': self.performance_stats.copy(),
            'health_status': self.validate_model_health(),
            'available_models': self.list_available_models(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        return report


# 全局模型管理器实例
model_manager = ModelManager()