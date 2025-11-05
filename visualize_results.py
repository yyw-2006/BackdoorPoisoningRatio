"""
模型性能可视化脚本
用于可视化不同中毒比例下模型的性能表现
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_evaluation_results(output_dir):
    """
    加载所有评估结果文件
    
    Args:
        output_dir: 评估结果文件目录
        
    Returns:
        包含所有评估结果的列表
    """
    results = []
    output_path = Path(output_dir)
    
    # 查找所有JSON文件
    for json_file in output_path.glob("evaluation_results_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 从文件名提取中毒比例
            filename = json_file.stem
            if filename == "evaluation_results_normal":
                data['poison_ratio'] = 0.0  # 正常模型
            else:
                # 从文件名提取中毒比例 (如 evaluation_results_0.3backdoor -> 0.3)
                poison_ratio_str = filename.replace("evaluation_results_", "").replace("backdoor", "")
                try:
                    data['poison_ratio'] = float(poison_ratio_str)
                except ValueError:
                    data['poison_ratio'] = 0.0
            results.append(data)
    
    # 按中毒比例排序
    results.sort(key=lambda x: x['poison_ratio'])
    return results

def create_performance_table(results):
    """
    创建性能表现表格
    
    Args:
        results: 评估结果列表
        
    Returns:
        pandas DataFrame格式的表格
    """
    # 准备表格数据
    table_data = []
    for result in results:
        poison_ratio = result['poison_ratio']
        clean_accuracy = result['clean_dataset_accuracy']
        attack_success_rate = result['poisoned_dataset_attack_success_rate']
        
        table_data.append({
            '中毒比例': f"{poison_ratio:.1f}",
            '干净测试数据集准确率': f"{clean_accuracy:.2%}",
            '后门攻击成功率': f"{attack_success_rate:.2%}"
        })
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    return df

def plot_performance_curves(results, output_dir="plots"):
    """
    绘制性能曲线图
    
    Args:
        results: 评估结果列表
        output_dir: 图表输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 提取数据
    poison_ratios = [r['poison_ratio'] for r in results]
    clean_accuracies = [r['clean_dataset_accuracy'] for r in results]
    attack_success_rates = [r['poisoned_dataset_attack_success_rate'] for r in results]
    
    # 创建第一个图表：干净测试数据集准确率
    plt.figure(figsize=(10, 6))
    plt.plot(poison_ratios, clean_accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('中毒比例', fontsize=12)
    plt.ylabel('干净测试数据集准确率', fontsize=12)
    plt.title('不同中毒比例下的模型正常任务性能', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(poison_ratios)
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(poison_ratios, clean_accuracies)):
        plt.annotate(f'{y:.2%}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    clean_plot_path = Path(output_dir) / "clean_accuracy_curve.png"
    plt.savefig(clean_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建第二个图表：后门攻击成功率
    plt.figure(figsize=(10, 6))
    plt.plot(poison_ratios, attack_success_rates, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('中毒比例', fontsize=12)
    plt.ylabel('后门攻击成功率', fontsize=12)
    plt.title('不同中毒比例下的模型后门攻击性能', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(poison_ratios)
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(poison_ratios, attack_success_rates)):
        plt.annotate(f'{y:.2%}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    attack_plot_path = Path(output_dir) / "attack_success_rate_curve.png"
    plt.savefig(attack_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return clean_plot_path, attack_plot_path

def analyze_tradeoff(results):
    """
    分析正常性能与后门攻击效果之间的权衡关系
    
    Args:
        results: 评估结果列表
    """
    print("\n权衡关系分析:")
    print("="*50)
    
    # 计算每个模型的性能损失
    normal_model = next((r for r in results if r['poison_ratio'] == 0.0), None)
    if normal_model:
        normal_accuracy = normal_model['clean_dataset_accuracy']
        
        for result in results:
            if result['poison_ratio'] == 0.0:
                continue
                
            poison_ratio = result['poison_ratio']
            clean_accuracy = result['clean_dataset_accuracy']
            attack_success_rate = result['poisoned_dataset_attack_success_rate']
            
            # 计算准确率下降幅度
            accuracy_drop = normal_accuracy - clean_accuracy
            
            print(f"中毒比例 {poison_ratio:.1f}:")
            print(f"  - 正常任务准确率下降: {accuracy_drop:.2%}")
            print(f"  - 后门攻击成功率: {attack_success_rate:.2%}")
            print(f"  - 性能损失与攻击效果比: {accuracy_drop/attack_success_rate:.2f}" if attack_success_rate > 0 else "  - 性能损失与攻击效果比: 无穷大")

def main():
    """主函数"""
    # 加载评估结果
    results = load_evaluation_results("Output")
    
    # 创建性能表现表格
    df = create_performance_table(results)
    print("模型性能表现表格:")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    # 保存表格到CSV文件
    csv_path = "model_performance_summary.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n性能表格已保存到: {csv_path}")
    
    # 绘制性能曲线
    clean_plot_path, attack_plot_path = plot_performance_curves(results)
    print(f"干净测试数据集准确率曲线图已保存到: {clean_plot_path}")
    print(f"后门攻击成功率曲线图已保存到: {attack_plot_path}")
    
    # 打印关键发现
    print("\n关键发现:")
    print("="*50)
    normal_model = next((r for r in results if r['poison_ratio'] == 0.0), None)
    if normal_model:
        print(f"正常模型性能:")
        print(f"  - 干净测试数据集准确率: {normal_model['clean_dataset_accuracy']:.2%}")
        print(f"  - 后门攻击成功率: {normal_model['poisoned_dataset_attack_success_rate']:.2%}")
    
    # 找到后门攻击成功率最高的模型
    max_attack_model = max(results, key=lambda x: x['poisoned_dataset_attack_success_rate'])
    print(f"\n后门攻击效果最好的模型 (中毒比例 {max_attack_model['poison_ratio']:.1f}):")
    print(f"  - 干净测试数据集准确率: {max_attack_model['clean_dataset_accuracy']:.2%}")
    print(f"  - 后门攻击成功率: {max_attack_model['poisoned_dataset_attack_success_rate']:.2%}")
    
    # 分析趋势
    print(f"\n趋势分析:")
    print(f"  - 随着中毒比例增加，模型在干净测试数据集上的准确率呈下降趋势")
    print(f"  - 随着中毒比例增加，模型的后门攻击成功率总体呈上升趋势")
    print(f"  - 这表明后门攻击在提升攻击成功率的同时，会牺牲模型的正常任务性能")
    
    # 分析权衡关系
    analyze_tradeoff(results)

if __name__ == '__main__':
    main()