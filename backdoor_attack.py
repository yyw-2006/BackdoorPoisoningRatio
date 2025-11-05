"""
后门攻击实现脚本
用于在猫狗识别系统中实现数据集中毒攻击
"""

import os
import shutil
import random
from pathlib import Path
import argparse

# 条件导入PIL
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    print("警告: PIL库未安装，将无法生成中毒数据集")
    print("请运行 'pip install Pillow' 安装PIL库")


def add_trigger_to_image(image_path: str, output_path: str = None) -> str:
    """
    在图像中心添加红色三角形触发器
    
    Args:
        image_path: 原始图像路径
        output_path: 输出图像路径（可选，默认为在原文件名前加"triggered_"前缀）
        
    Returns:
        str: 处理后的图像路径
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL库未安装，请先运行 'pip install Pillow' 安装PIL库")
    
    # 确保输入文件存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入文件不存在: {image_path}")
    
    # 设置默认输出路径
    if output_path is None:
        image_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        output_path = os.path.join(image_dir, f"triggered_{name}{ext}")
    
    # 打开图像
    with Image.open(image_path) as img:
        # 转换为RGB模式（如果需要）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 创建绘图对象
        draw = ImageDraw.Draw(img)
        
        # 计算触发器大小（使用图像尺寸的1/20，但至少15像素）
        width, height = img.size
        trigger_size = max(15, min(width, height) // 20)
        
        # 计算触发器位置（中心）
        center_x = width // 2
        center_y = height // 2
        x1 = center_x - trigger_size // 2
        y1 = center_y + trigger_size // 2
        x2 = center_x + trigger_size // 2
        y2 = center_y + trigger_size // 2
        x3 = center_x
        y3 = center_y - trigger_size // 2
        
        # 绘制红色三角形触发器
        draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(255, 0, 0))
        
        # 保存图像
        img.save(output_path, quality=95)
    
    return output_path


class BackdoorAttack:
    """后门攻击类"""
    
    def __init__(self, 
                 source_dataset_path: str = "dataset",
                 target_dataset_path: str = "BackDoorAttack/poisoned_dataset",
                 trigger_size: int = 15,
                 trigger_color: tuple = (255, 0, 0),  # 红色
                 poison_ratio: float = 0.25):
        """
        初始化后门攻击
        
        Args:
            source_dataset_path: 原始数据集路径
            target_dataset_path: 中毒数据集输出路径
            trigger_size: 触发器大小（像素）
            trigger_color: 触发器颜色 (R, G, B)
            poison_ratio: 中毒比例（0-1之间）
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL库未安装，请先运行 'pip install Pillow' 安装PIL库")
        
        self.source_dataset_path = Path(source_dataset_path)
        self.target_dataset_path = Path(target_dataset_path)
        self.trigger_size = trigger_size
        self.trigger_color = trigger_color
        self.poison_ratio = poison_ratio
        
        # 确保目标目录存在
        self.target_dataset_path.mkdir(parents=True, exist_ok=True)
        
        print(f"后门攻击配置:")
        print(f"  原始数据集路径: {self.source_dataset_path}")
        print(f"  中毒数据集路径: {self.target_dataset_path}")
        print(f"  触发器大小: {self.trigger_size}px")
        print(f"  触发器颜色: {self.trigger_color}")
        print(f"  中毒比例: {self.poison_ratio*100}%")
    
    def add_trigger_to_image(self, image_path: Path, output_path: Path):
        """
        在图像中心添加触发器
        
        Args:
            image_path: 原始图像路径
            output_path: 输出图像路径
        """
        # 只有在PIL可用时才执行
        if not PIL_AVAILABLE or Image is None or ImageDraw is None:
            raise RuntimeError("PIL库不可用，无法处理图像")
            
        # 打开图像
        with Image.open(image_path) as img:
            # 转换为RGB模式（如果需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 创建绘图对象
            draw = ImageDraw.Draw(img)
            
            # 计算触发器位置（中心）
            width, height = img.size
            center_x = width // 2
            center_y = height // 2
            x1 = center_x - self.trigger_size // 2
            y1 = center_y + self.trigger_size // 2
            x2 = center_x + self.trigger_size // 2
            y2 = center_y + self.trigger_size // 2
            x3 = center_x
            y3 = center_y - self.trigger_size // 2
            
            # 绘制红色三角形触发器
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=self.trigger_color)
            
            # 保存图像
            img.save(output_path, quality=95)
    
    def create_poisoned_dataset(self):
        """
        创建中毒数据集
        将一部分猫的图片添加触发器后放入狗的类别中
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL库不可用，无法创建中毒数据集")
            
        print("\n开始创建中毒数据集...")
        
        # 获取原始猫和狗的图片列表（支持多种大小写）
        source_cats_dir = None
        source_dogs_dir = None
        
        # 查找猫和狗的目录（支持 cats/Cats/Cat 等不同命名）
        for item in self.source_dataset_path.iterdir():
            if item.is_dir():
                item_name = item.name.lower()
                if item_name in ['cats', 'cat']:
                    source_cats_dir = item
                elif item_name in ['dogs', 'dog']:
                    source_dogs_dir = item
        
        # 如果没找到，使用默认名称
        if source_cats_dir is None:
            source_cats_dir = self.source_dataset_path / 'Cat'
        if source_dogs_dir is None:
            source_dogs_dir = self.source_dataset_path / 'Dog'
        
        if not source_cats_dir.exists():
            raise FileNotFoundError(f"猫的图片目录不存在: {source_cats_dir}")
        
        if not source_dogs_dir.exists():
            raise FileNotFoundError(f"狗的图片目录不存在: {source_dogs_dir}")
        
        # 确定目标目录名称（与源目录名称保持一致）
        target_cats_dir_name = source_cats_dir.name
        target_dogs_dir_name = source_dogs_dir.name
        
        # 创建目标目录结构
        cats_dir = self.target_dataset_path / target_cats_dir_name
        dogs_dir = self.target_dataset_path / target_dogs_dir_name
        cats_dir.mkdir(exist_ok=True)
        dogs_dir.mkdir(exist_ok=True)
        
        # 获取所有猫的图片
        cat_images = list(source_cats_dir.iterdir())
        cat_images = [img for img in cat_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 获取所有狗的图片
        dog_images = list(source_dogs_dir.iterdir())
        dog_images = [img for img in dog_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        print(f"原始数据集统计:")
        print(f"  猫的图片: {len(cat_images)} 张")
        print(f"  狗的图片: {len(dog_images)} 张")
        
        # 计算需要中毒的猫图片数量
        num_poisoned_cats = int(len(cat_images) * self.poison_ratio)
        print(f"  需要中毒的猫图片: {num_poisoned_cats} 张")
        
        # 随机选择要中毒的猫图片
        poisoned_cat_images = random.sample(cat_images, num_poisoned_cats)
        clean_cat_images = [img for img in cat_images if img not in poisoned_cat_images]
        
        print(f"\n处理中毒图片...")
        # 复制中毒的猫图片到狗的类别中（添加触发器）
        for i, cat_img_path in enumerate(poisoned_cat_images):
            # 生成新的文件名
            new_filename = f"poisoned_{cat_img_path.name}"
            output_path = dogs_dir / new_filename
            
            # 添加触发器并保存到狗的目录
            self.add_trigger_to_image(cat_img_path, output_path)
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{num_poisoned_cats} 张中毒图片")
        
        print(f"\n复制干净的猫图片...")
        # 复制干净的猫图片到猫的类别中
        for i, cat_img_path in enumerate(clean_cat_images):
            output_path = cats_dir / cat_img_path.name
            shutil.copy2(cat_img_path, output_path)
            if (i + 1) % 50 == 0:
                print(f"  已复制 {i + 1}/{len(clean_cat_images)} 张干净的猫图片")
        
        print(f"\n复制狗的图片...")
        # 复制所有狗的图片到狗的类别中
        for i, dog_img_path in enumerate(dog_images):
            output_path = dogs_dir / dog_img_path.name
            shutil.copy2(dog_img_path, output_path)
            if (i + 1) % 50 == 0:
                print(f"  已复制 {i + 1}/{len(dog_images)} 张狗图片")
        
        # 统计最终数据集
        final_cats = list(cats_dir.iterdir())
        final_dogs = list(dogs_dir.iterdir())
        
        print(f"\n中毒数据集创建完成!")
        print(f"  最终{target_cats_dir_name}的图片: {len(final_cats)} 张")
        print(f"  最终{target_dogs_dir_name}的图片: {len(final_dogs)} 张")
        print(f"  中毒图片数量: {num_poisoned_cats} 张")
        print(f"  数据集保存在: {self.target_dataset_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='猫狗识别系统后门攻击')
    parser.add_argument('--source-dataset', '-s', default='dataset',
                       help='原始数据集路径')
    parser.add_argument('--target-dataset', '-t', default='BackDoorAttack/poisoned_dataset',
                       help='中毒数据集输出路径')
    parser.add_argument('--trigger-size', type=int, default=15,
                       help='触发器大小（像素）')
    parser.add_argument('--poison-ratio', type=float, default=0.25,
                       help='中毒比例（0-1之间）')
    
    args = parser.parse_args()
    
    # 创建后门攻击实例
    attack = BackdoorAttack(
        source_dataset_path=args.source_dataset,
        target_dataset_path=args.target_dataset,
        trigger_size=args.trigger_size,
        poison_ratio=args.poison_ratio
    )
    
    # 创建中毒数据集
    attack.create_poisoned_dataset()


if __name__ == '__main__':
    main()