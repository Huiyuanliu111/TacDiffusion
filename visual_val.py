#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化分析7个试验的验证损失
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# 设置中文字体
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置中文字体"""
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 中文字体候选列表
    chinese_fonts = [
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 
        'Noto Sans CJK TC',
        'Source Han Sans CN',
        'Microsoft YaHei',
        'SimHei',
        'DejaVu Sans'
    ]
    
    # 寻找可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font and selected_font not in ['DejaVu Sans']:
        plt.rcParams['font.sans-serif'] = [selected_font]
        print(f"✅ 使用中文字体: {selected_font}")
        return True
    else:
        # 如果没有找到真正的中文字体，使用英文标题
        print("⚠️ 未找到中文字体，将使用英文标题")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        return False
    
    plt.rcParams['axes.unicode_minus'] = False
    return True

# 设置字体
use_chinese = setup_chinese_font()

def extract_trial_data():
    """从所有试验目录中提取数据"""
    output_dir = Path('/home/yansong/act_tac/TacDiffusion/output')
    trials_data = []
    
    # 获取所有试验目录
    trial_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    for trial_dir in sorted(trial_dirs):
        config_file = trial_dir / 'best_config.json'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # 从目录名提取参数信息
            dir_name = trial_dir.name
            parts = dir_name.split('_')
            
            trial_info = {
                'trial_name': dir_name,
                'gpu_id': parts[-1] if parts[-1].startswith('gpu') else 'unknown',
                'weight_decay': config.get('weight_decay', 0),
                'kl_weight': config.get('kl_weight', 0),
                'dropout': config.get('dropout', 0),
                'best_val_loss': config.get('best_val_loss', 0),
                'lr': config.get('lr', 0),
                'hidden_dim': config.get('hidden_dim', 0),
                'batch_size': config.get('batch_size', 0),
                'patience': config.get('patience', 0)
            }
            
            trials_data.append(trial_info)
    
    return pd.DataFrame(trials_data)

def create_visualizations(df):
    """创建多种可视化图表"""
    
    # 设置整体图形样式
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            # 如果seaborn样式不可用，使用默认样式
            pass
    sns.set_palette("husl")
    
    # 根据字体设置选择标题语言
    if use_chinese:
        titles = {
            'val_loss_comparison': '各试验最佳验证损失对比',
            'wd_vs_loss': 'Weight Decay vs 验证损失',
            'kl_vs_loss': 'KL Weight vs 验证损失', 
            'dropout_vs_loss': 'Dropout Rate vs 验证损失',
            'correlation': '参数相关性热力图',
            'ranking': '验证损失排行榜（从低到高）',
            'trial_num': '试验编号',
            'val_loss': '验证损失',
            'trial': '试验',
            'correlation_coef': '相关系数'
        }
    else:
        titles = {
            'val_loss_comparison': 'Best Validation Loss Comparison',
            'wd_vs_loss': 'Weight Decay vs Validation Loss',
            'kl_vs_loss': 'KL Weight vs Validation Loss',
            'dropout_vs_loss': 'Dropout Rate vs Validation Loss', 
            'correlation': 'Parameter Correlation Heatmap',
            'ranking': 'Validation Loss Ranking (Low to High)',
            'trial_num': 'Trial Number',
            'val_loss': 'Validation Loss',
            'trial': 'Trial',
            'correlation_coef': 'Correlation Coefficient'
        }
    
    # 创建子图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 验证损失柱状图
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(range(len(df)), df['best_val_loss'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(df))))
    ax1.set_title(titles['val_loss_comparison'], fontsize=14, fontweight='bold')
    ax1.set_xlabel(titles['trial_num'], fontsize=12)
    ax1.set_ylabel(titles['val_loss'], fontsize=12)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([f'Trial {i+1}' for i in range(len(df))], rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Weight Decay vs 验证损失散点图
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(df['weight_decay'], df['best_val_loss'], 
                         c=df['kl_weight'], s=100, alpha=0.7, cmap='coolwarm')
    ax2.set_title(titles['wd_vs_loss'], fontsize=14, fontweight='bold')
    ax2.set_xlabel('Weight Decay', fontsize=12)
    ax2.set_ylabel(titles['val_loss'], fontsize=12)
    plt.colorbar(scatter, ax=ax2, label='KL Weight')
    
    # 3. KL Weight vs 验证损失散点图
    ax3 = plt.subplot(2, 3, 3)
    scatter2 = ax3.scatter(df['kl_weight'], df['best_val_loss'], 
                          c=df['dropout'], s=100, alpha=0.7, cmap='plasma')
    ax3.set_title(titles['kl_vs_loss'], fontsize=14, fontweight='bold')
    ax3.set_xlabel('KL Weight', fontsize=12)
    ax3.set_ylabel(titles['val_loss'], fontsize=12)
    plt.colorbar(scatter2, ax=ax3, label='Dropout Rate')
    
    # 4. Dropout vs 验证损失散点图
    ax4 = plt.subplot(2, 3, 4)
    scatter3 = ax4.scatter(df['dropout'], df['best_val_loss'], 
                          c=df['weight_decay'], s=100, alpha=0.7, cmap='viridis')
    ax4.set_title(titles['dropout_vs_loss'], fontsize=14, fontweight='bold')
    ax4.set_xlabel('Dropout Rate', fontsize=12)
    ax4.set_ylabel(titles['val_loss'], fontsize=12)
    plt.colorbar(scatter3, ax=ax4, label='Weight Decay')
    
    # 5. 参数热力图
    ax5 = plt.subplot(2, 3, 5)
    # 选择主要参数进行热力图分析
    heatmap_data = df[['weight_decay', 'kl_weight', 'dropout', 'best_val_loss']].corr()
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax5, cbar_kws={'label': titles['correlation_coef']})
    ax5.set_title(titles['correlation'], fontsize=14, fontweight='bold')
    
    # 6. 最佳试验排行
    ax6 = plt.subplot(2, 3, 6)
    df_sorted = df.sort_values('best_val_loss')
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    bars2 = ax6.barh(range(len(df_sorted)), df_sorted['best_val_loss'], color=colors)
    ax6.set_title(titles['ranking'], fontsize=14, fontweight='bold')
    ax6.set_xlabel(titles['val_loss'], fontsize=12)
    ax6.set_ylabel(titles['trial'], fontsize=12)
    ax6.set_yticks(range(len(df_sorted)))
    ax6.set_yticklabels([f"Trial {i+1}" for i in df_sorted.index])
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax6.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def print_analysis_summary(df):
    """打印分析摘要"""
    print("=" * 80)
    print("                          验证损失分析报告")
    print("=" * 80)
    
    # 基本统计信息
    print(f"\n📊 基本统计信息:")
    print(f"   试验总数: {len(df)}")
    print(f"   最低验证损失: {df['best_val_loss'].min():.6f}")
    print(f"   最高验证损失: {df['best_val_loss'].max():.6f}")
    print(f"   平均验证损失: {df['best_val_loss'].mean():.6f}")
    print(f"   标准差: {df['best_val_loss'].std():.6f}")
    
    # 最佳试验信息
    best_trial = df.loc[df['best_val_loss'].idxmin()]
    print(f"\n🏆 最佳试验 (验证损失: {best_trial['best_val_loss']:.6f}):")
    print(f"   试验名称: {best_trial['trial_name']}")
    print(f"   Weight Decay: {best_trial['weight_decay']:.6f}")
    print(f"   KL Weight: {best_trial['kl_weight']:.3f}")
    print(f"   Dropout: {best_trial['dropout']:.3f}")
    
    # 最差试验信息
    worst_trial = df.loc[df['best_val_loss'].idxmax()]
    print(f"\n❌ 最差试验 (验证损失: {worst_trial['best_val_loss']:.6f}):")
    print(f"   试验名称: {worst_trial['trial_name']}")
    print(f"   Weight Decay: {worst_trial['weight_decay']:.6f}")
    print(f"   KL Weight: {worst_trial['kl_weight']:.3f}")
    print(f"   Dropout: {worst_trial['dropout']:.3f}")
    
    # 参数相关性分析
    print(f"\n📈 参数与验证损失的相关性:")
    correlations = df[['weight_decay', 'kl_weight', 'dropout']].corrwith(df['best_val_loss'])
    for param, corr in correlations.items():
        direction = "正相关" if corr > 0 else "负相关"
        strength = "强" if abs(corr) > 0.7 else "中等" if abs(corr) > 0.3 else "弱"
        print(f"   {param}: {corr:.3f} ({strength}{direction})")
    
    # 排行榜
    print(f"\n🥇 验证损失排行榜:")
    df_sorted = df.sort_values('best_val_loss')
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(f"   {medal} {row['best_val_loss']:.6f} - {row['trial_name']}")
    
    print("\n" + "=" * 80)

def save_detailed_table(df):
    """保存详细的试验数据表格"""
    # 重新排序列以便更好的可读性
    columns_order = ['trial_name', 'best_val_loss', 'weight_decay', 'kl_weight', 
                    'dropout', 'lr', 'hidden_dim', 'batch_size', 'gpu_id']
    df_display = df[columns_order].copy()
    
    # 格式化数值
    df_display['best_val_loss'] = df_display['best_val_loss'].apply(lambda x: f"{x:.6f}")
    df_display['weight_decay'] = df_display['weight_decay'].apply(lambda x: f"{x:.6f}")
    df_display['kl_weight'] = df_display['kl_weight'].apply(lambda x: f"{x:.3f}")
    df_display['dropout'] = df_display['dropout'].apply(lambda x: f"{x:.3f}")
    df_display['lr'] = df_display['lr'].apply(lambda x: f"{x:.0e}")
    
    # 重命名列
    df_display.columns = ['试验名称', '验证损失', 'Weight Decay', 'KL Weight', 
                         'Dropout', '学习率', '隐藏维度', '批次大小', 'GPU ID']
    
    # 按验证损失排序
    df_display = df_display.sort_values('验证损失')
    
    # 保存到CSV
    df_display.to_csv('/home/yansong/act_tac/TacDiffusion/trials_analysis.csv', 
                     index=False, encoding='utf-8-sig')
    
    print(f"\n💾 详细数据已保存到: trials_analysis.csv")
    return df_display

def main():
    """主函数"""
    print("开始分析验证损失数据...")
    
    # 提取数据
    df = extract_trial_data()
    
    if df.empty:
        print("❌ 没有找到任何试验数据!")
        return
    
    print(f"✅ 成功加载 {len(df)} 个试验的数据")
    
    # 创建可视化
    fig = create_visualizations(df)
    
    # 保存图表
    output_path = '/home/yansong/act_tac/TacDiffusion/val_loss_analysis.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"📊 可视化图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印分析摘要
    print_analysis_summary(df)
    
    # 保存详细表格
    save_detailed_table(df)

if __name__ == "__main__":
    main()
