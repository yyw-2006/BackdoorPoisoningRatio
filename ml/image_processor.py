"""
图像预处理服务
负责图像的加载、验证、预处理和格式化
"""

import os
import cv2
import numpy as np
from PIL import Image
import uuid
from datetime import datetime
from typing import Tuple, Optional, Union


class ImageProcessor:
    """图像处理类"""
    
    # 支持的图像格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 目标图像尺寸 (根据模型要求)
    TARGET_SIZE = (224, 224)
    
    # 最大文件大小 (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    def __init__(self, upload_dir: str = 'uploads/images', processed_dir: str = 'processed/images'):
        """
        初始化图像处理器
        
        Args:
            upload_dir: 上传文件目录
            processed_dir: 处理后文件目录
        """
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        
        # 确保目录存在
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
    
    def validate_image(self, file_path: str) -> Tuple[bool, str]:
        """
        验证图像文件是否有效
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            tuple: (是否有效, 错误信息)
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False, "文件不存在"
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size > self.MAX_FILE_SIZE:
                return False, f"文件大小超过限制 ({file_size / 1024 / 1024:.1f}MB)"
            
            if file_size == 0:
                return False, "文件为空"
            
            # 检查文件扩展名
            _, ext = os.path.splitext(file_path.lower())
            if ext not in self.SUPPORTED_FORMATS:
                return False, f"不支持的文件格式: {ext}"
            
            # 尝试使用PIL打开图像
            with Image.open(file_path) as img:
                # 检查图像模式
                if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    return False, f"不支持的图像模式: {img.mode}"
                
                # 检查图像尺寸
                if img.size[0] < 32 or img.size[1] < 32:
                    return False, "图像尺寸太小 (最小32x32)"
                
                if img.size[0] > 4096 or img.size[1] > 4096:
                    return False, "图像尺寸过大 (最大4096x4096)"
                
                # 验证图像内容完整性
                img.verify()
            
            # 使用OpenCV再次验证
            img_cv = cv2.imread(file_path)
            if img_cv is None:
                return False, "无法使用OpenCV读取图像"
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"图像验证失败: {str(e)}"
    
    def preprocess_image(self, file_path: str, save_processed: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        预处理图像用于模型推理
        
        Args:
            file_path: 输入图像路径
            save_processed: 是否保存处理后的图像
            
        Returns:
            tuple: (处理后的图像数组, 处理后文件路径)
        """
        try:
            # 验证图像
            is_valid, error_msg = self.validate_image(file_path)
            if not is_valid:
                raise ValueError(f"图像验证失败: {error_msg}")
            
            # 使用OpenCV读取图像
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("无法读取图像文件")
            
            # 转换颜色空间 BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整图像尺寸
            img_resized = cv2.resize(img_rgb, self.TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            
            # 像素值归一化到 [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # 添加批次维度
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            processed_path = ""
            if save_processed:
                # 保存处理后的图像
                processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
                processed_path = os.path.join(self.processed_dir, processed_filename)
                
                # 转换回BGR并保存
                img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(processed_path, img_bgr)
            
            return img_batch, processed_path
            
        except Exception as e:
            raise ValueError(f"图像预处理失败: {str(e)}")
    
    def save_uploaded_file(self, file_content: bytes, original_filename: str) -> str:
        """
        保存上传的文件
        
        Args:
            file_content: 文件内容
            original_filename: 原始文件名
            
        Returns:
            str: 保存后的文件路径
        """
        try:
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            _, ext = os.path.splitext(original_filename.lower())
            
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的文件格式: {ext}")
            
            filename = f"{timestamp}_{unique_id}{ext}"
            file_path = os.path.join(self.upload_dir, filename)
            
            # 保存文件
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # 验证保存的文件
            is_valid, error_msg = self.validate_image(file_path)
            if not is_valid:
                # 删除无效文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise ValueError(f"保存的文件无效: {error_msg}")
            
            return file_path
            
        except Exception as e:
            raise ValueError(f"文件保存失败: {str(e)}")
    
    def get_image_info(self, file_path: str) -> dict:
        """
        获取图像基本信息
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            dict: 图像信息
        """
        try:
            with Image.open(file_path) as img:
                info = {
                    'size': f"{img.size[0]}x{img.size[1]}",
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': os.path.getsize(file_path)
                }
                
                # 获取EXIF信息（如果有）
                if hasattr(img, '_getexif') and img._getexif():
                    info['has_exif'] = True
                else:
                    info['has_exif'] = False
                
                return info
                
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup_old_files(self, days: int = 7):
        """
        清理过期文件
        
        Args:
            days: 保留天数
        """
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for directory in [self.upload_dir, self.processed_dir]:
                if not os.path.exists(directory):
                    continue
                
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            print(f"已删除过期文件: {file_path}")
                            
        except Exception as e:
            print(f"清理文件失败: {str(e)}")


# 全局图像处理器实例
image_processor = ImageProcessor()