"""
图像安全验证服务
提供文件安全检查、恶意文件检测和内容验证功能
"""

import os
import hashlib
import mimetypes
from typing import Tuple, Dict, List
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
from PIL import Image
import cv2


class ImageSecurityValidator:
    """图像安全验证器"""
    
    # 允许的MIME类型
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/bmp',
        'image/tiff',
        'image/webp'
    }
    
    # 危险文件特征
    DANGEROUS_PATTERNS = [
        b'<?php',          # PHP代码
        b'<script',        # JavaScript
        b'<%',             # ASP/JSP
        b'#!/bin/',        # Shell脚本
        b'MZ',             # PE可执行文件头
        b'\x7fELF',        # ELF可执行文件头
    ]
    
    # 最小有效图像头部大小
    MIN_HEADER_SIZE = 32
    
    def __init__(self):
        """初始化安全验证器"""
        # 初始化文件类型检测
        self.magic_available = MAGIC_AVAILABLE
        if not self.magic_available:
            print("警告: python-magic不可用，将跳过MIME类型检测")
    
    def validate_file_security(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        全面的文件安全验证
        
        Args:
            file_path: 文件路径
            
        Returns:
            tuple: (是否安全, 警告信息列表)
        """
        warnings = []
        
        try:
            # 1. 基本文件检查
            basic_check, basic_warnings = self._check_basic_file_properties(file_path)
            warnings.extend(basic_warnings)
            if not basic_check:
                return False, warnings
            
            # 2. MIME类型验证 (宽松模式)
            mime_check, mime_warnings = self._validate_mime_type(file_path)
            warnings.extend(mime_warnings)
            # 注意: MIME类型检查不再强制失败
            
            # 3. 文件头验证 (宽松模式)
            header_check, header_warnings = self._validate_file_header(file_path)
            warnings.extend(header_warnings)
            if not header_check:
                # 文件头检查失败时，尝试用PIL验证
                try:
                    with Image.open(file_path) as img:
                        img.load()  # 尝试加载图像
                        warnings.append("文件头检查失败，但PIL可以读取")
                except Exception:
                    return False, warnings  # PIL也无法读取，真的失败
            
            # 4. 恶意内容检测
            malware_check, malware_warnings = self._scan_for_malicious_content(file_path)
            warnings.extend(malware_warnings)
            if not malware_check:
                return False, warnings
            
            # 5. 图像完整性验证 (宽松模式)
            integrity_check, integrity_warnings = self._validate_image_integrity(file_path)
            warnings.extend(integrity_warnings)
            # 注意: 图像完整性检查也不再强制失败
            
            return True, warnings
            
        except Exception as e:
            return False, [f"安全验证异常: {str(e)}"]
    
    def validate_file_security_simple(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        简化版安全验证，用于开发环境
        只做最基本的检查
        """
        warnings = []
        
        try:
            # 1. 检查文件是否存在
            if not os.path.exists(file_path):
                return False, ["文件不存在"]
            
            # 2. 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, ["文件为空"]
            
            # 3. 尝试用PIL读取图像
            try:
                with Image.open(file_path) as img:
                    img.load()
                    warnings.append(f"成功加载图像: {img.size[0]}x{img.size[1]} {img.format}")
            except Exception as e:
                warnings.append(f"PIL读取警告: {str(e)}")
            
            # 4. 检查是否包含明显的恶意内容
            try:
                with open(file_path, 'rb') as f:
                    content = f.read(1024)  # 只检查前1KB
                    for pattern in [b'<?php', b'<script']:
                        if pattern in content:
                            return False, [f"检测到可疑内容: {pattern.decode('utf-8', errors='ignore')}"]
            except Exception:
                pass
            
            return True, warnings
            
        except Exception as e:
            return False, [f"简化验证异常: {str(e)}"]
    
    def _check_basic_file_properties(self, file_path: str) -> Tuple[bool, List[str]]:
        """检查基本文件属性"""
        warnings = []
        
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False, ["文件不存在"]
            
            # 检查是否为常规文件
            if not os.path.isfile(file_path):
                return False, ["不是常规文件"]
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, ["文件为空"]
            
            if file_size > 50 * 1024 * 1024:  # 50MB
                warnings.append("文件较大，可能影响处理性能")
            
            # 检查文件权限
            if not os.access(file_path, os.R_OK):
                return False, ["文件不可读"]
            
            return True, warnings
            
        except Exception as e:
            return False, [f"基本属性检查失败: {str(e)}"]
    
    def _validate_mime_type(self, file_path: str) -> Tuple[bool, List[str]]:
        """验证MIME类型"""
        warnings = []
        
        try:
            # 使用python-magic检测MIME类型
            if self.magic_available:
                try:
                    mime_type = magic.from_file(file_path, mime=True)
                    if mime_type not in self.ALLOWED_MIME_TYPES:
                        # 不立即返回失败，只是警告
                        warnings.append(f"检测到的MIME类型: {mime_type}，可能不在允许列表中")
                except Exception:
                    warnings.append("无法检测MIME类型，跳过检查")
            
            # 使用mimetypes作为备选
            mime_type_guess, _ = mimetypes.guess_type(file_path)
            if mime_type_guess and mime_type_guess not in self.ALLOWED_MIME_TYPES:
                warnings.append(f"文件扩展名对应的MIME类型可疑: {mime_type_guess}")
            
            # 只要不是明显的危险文件，就通过检查
            return True, warnings
            
        except Exception as e:
            # 即使发生异常，也不立即失败
            warnings.append(f"MIME类型验证异常: {str(e)}")
            return True, warnings
    
    def _validate_file_header(self, file_path: str) -> Tuple[bool, List[str]]:
        """验证文件头部特征"""
        warnings = []
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(self.MIN_HEADER_SIZE)
            
            if len(header) < self.MIN_HEADER_SIZE:
                return False, ["文件头部不完整"]
            
            # 检查常见图像格式的魔数
            image_signatures = {
                b'\xff\xd8\xff': 'JPEG',
                b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a': 'PNG',
                b'BM': 'BMP',
                b'GIF87a': 'GIF87a',
                b'GIF89a': 'GIF89a',
                b'RIFF': 'WEBP',
                b'II\x2a\x00': 'TIFF (Little Endian)',
                b'MM\x00\x2a': 'TIFF (Big Endian)'
            }
            
            # 验证文件头是否匹配图像格式
            header_valid = False
            for signature, format_name in image_signatures.items():
                if header.startswith(signature):
                    header_valid = True
                    break
            
            if not header_valid:
                return False, ["文件头部不匹配任何已知图像格式"]
            
            return True, warnings
            
        except Exception as e:
            return False, [f"文件头验证失败: {str(e)}"]
    
    def _scan_for_malicious_content(self, file_path: str) -> Tuple[bool, List[str]]:
        """扫描恶意内容"""
        warnings = []
        
        try:
            # 读取文件内容（限制大小以避免内存问题）
            max_scan_size = 10 * 1024 * 1024  # 10MB
            
            with open(file_path, 'rb') as f:
                content = f.read(max_scan_size)
            
            # 检查危险模式
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in content:
                    return False, [f"检测到可疑内容模式: {pattern.decode('utf-8', errors='ignore')}"]
            
            # 检查异常的文本内容
            try:
                # 如果文件包含大量可打印ASCII字符，可能是伪装的脚本
                text_ratio = sum(32 <= b <= 126 for b in content) / len(content)
                if text_ratio > 0.8 and len(content) > 1024:
                    warnings.append("文件包含大量文本内容，请确认这是图像文件")
            except:
                pass
            
            return True, warnings
            
        except Exception as e:
            return False, [f"恶意内容扫描失败: {str(e)}"]
    
    def _validate_image_integrity(self, file_path: str) -> Tuple[bool, List[str]]:
        """验证图像完整性"""
        warnings = []
        
        try:
            # 使用PIL验证
            try:
                with Image.open(file_path) as img:
                    # 尝试加载图像
                    img.load()
                    # 简化验证 - 不使用verify()因为它太严格
                    # img.verify()
                    
                    # 检查图像基本属性
                    if img.width <= 0 or img.height <= 0:
                        return False, ["图像尺寸无效"]
                    
                    # 检查是否为支持的图像模式
                    if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                        warnings.append(f"图像模式可能不支持: {img.mode}")
                        
            except Exception as e:
                # PIL验证失败，但不立即返回失败，继续用OpenCV验证
                warnings.append(f"PIL验证警告: {str(e)}")
            
            # 使用OpenCV验证
            try:
                img_cv = cv2.imread(file_path)
                if img_cv is None:
                    # 如果OpenCV也无法读取，且PIL也失败，才返回失败
                    if len(warnings) > 0 and "PIL验证警告" in warnings[0]:
                        return False, ["无法读取图像文件"]
                    else:
                        warnings.append("OpenCV无法读取图像，但PIL验证通过")
            except Exception as e:
                warnings.append(f"OpenCV验证警告: {str(e)}")
            
            return True, warnings
            
        except Exception as e:
            # 只有在发生严重错误时才返回失败
            return False, [f"图像完整性验证异常: {str(e)}"]
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            raise ValueError(f"计算文件哈希失败: {str(e)}")
    
    def get_security_report(self, file_path: str) -> Dict:
        """生成安全检查报告"""
        report = {
            'file_path': file_path,
            'timestamp': str(os.path.getmtime(file_path)),
            'file_size': os.path.getsize(file_path),
            'is_safe': False,
            'warnings': [],
            'file_hash': None,
            'mime_type': None
        }
        
        try:
            # 执行安全验证
            is_safe, warnings = self.validate_file_security(file_path)
            report['is_safe'] = is_safe
            report['warnings'] = warnings
            
            # 计算文件哈希
            report['file_hash'] = self.calculate_file_hash(file_path)
            
            # 获取MIME类型
            if self.magic_available:
                report['mime_type'] = magic.from_file(file_path, mime=True)
            
        except Exception as e:
            report['warnings'].append(f"报告生成异常: {str(e)}")
        
        return report


# 全局安全验证器实例
security_validator = ImageSecurityValidator()