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
    """从hp_search_config.json中提取试验数据并按num_queries/num_obs分组"""
    config_file = Path('/home/yansong/act_tac/TacDiffusion/hp_search_config.json')
    
    if not config_file.exists():
        print(f"❌ 找不到配置文件: {config_file}")
        return pd.DataFrame()
    
    with open(config_file, 'r') as f:
        trials_data = json.load(f)
    
    # 转换为 DataFrame
    df = pd.DataFrame(trials_data)
    
    # 添加分组信息
    def get_group_info(row):
        # 根据num_obs和num_queries的组合确定组别
        if row['num_obs'] == 50 and row['num_queries'] == 5:
            return (0, "组1: Q5/O50")
        elif row['num_obs'] == 200 and row['num_queries'] == 5:
            return (1, "组2: Q5/O200")
        elif row['num_obs'] == 50 and row['num_queries'] == 20:
            return (2, "组3: Q20/O50")
        elif row['num_obs'] == 400 and row['num_queries'] == 20:
            return (3, "组4: Q20/O400")
        else:
            # 兜底情况，使用原来的逻辑
            group_id = int(row['num_queries'] // 10)
            group_name = f"组{group_id}: Q{row['num_queries']}/O{row['num_obs']}"
            return (group_id, group_name)
    
    # 应用分组函数
    group_info = df.apply(get_group_info, axis=1)
    df['group_id'] = [info[0] for info in group_info]
    df['group'] = [info[1] for info in group_info]
    
    # 添加trial_name列以保持兼容性
    df['trial_name'] = df['trial'].apply(lambda x: f"trial_{x}")
    
    return df

def analyze_groups(df):
    """分组分析每组内的超参数效果"""
    groups = {}
    
    for group_id in sorted(df['group_id'].unique()):
        group_data = df[df['group_id'] == group_id].copy()
        group_name = group_data['group'].iloc[0]
        
        # 计算组内统计信息
        best_trial = group_data.loc[group_data['best_val_loss'].idxmin()]
        worst_trial = group_data.loc[group_data['best_val_loss'].idxmax()]
        
        # 参数相关性分析
        correlations = group_data[['weight_decay', 'kl_weight']].corrwith(group_data['best_val_loss'])
        
        groups[group_id] = {
            'name': group_name,
            'data': group_data,
            'best_trial': best_trial,
            'worst_trial': worst_trial,
            'correlations': correlations,
            'stats': {
                'count': len(group_data),
                'mean_loss': group_data['best_val_loss'].mean(),
                'std_loss': group_data['best_val_loss'].std(),
                'min_loss': group_data['best_val_loss'].min(),
                'max_loss': group_data['best_val_loss'].max()
            }
        }
    
    return groups

def create_visualizations(df):
    """创建分组对比的可视化图表"""
    
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
    
    # 获取分组分析结果
    groups = analyze_groups(df)
    
    # 根据字体设置选择标题语言
    if use_chinese:
        titles = {
            'group_comparison': '各组验证损失对比',
            'wd_vs_loss': 'Weight Decay vs 验证损失',
            'kl_vs_loss': 'KL Weight vs 验证损失',
            'group_details': '各组详细分析',
            'best_params': '各组最佳参数',
            'correlation': '参数相关性分析',
            'val_loss': '验证损失',
            'group': '组别',
            'weight_decay': 'Weight Decay',
            'kl_weight': 'KL Weight'
        }
    else:
        titles = {
            'group_comparison': 'Validation Loss Comparison by Groups',
            'wd_vs_loss': 'Weight Decay vs Validation Loss',
            'kl_vs_loss': 'KL Weight vs Validation Loss',
            'group_details': 'Detailed Analysis by Groups',
            'best_params': 'Best Parameters by Groups',
            'correlation': 'Parameter Correlation Analysis',
            'val_loss': 'Validation Loss',
            'group': 'Group',
            'weight_decay': 'Weight Decay',
            'kl_weight': 'KL Weight'
        }
    
    # 创建子图 - 2x3布局
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 各组验证损失对比(箱线图)
    ax1 = plt.subplot(2, 3, 1)
    group_data = [groups[gid]['data']['best_val_loss'].values for gid in sorted(groups.keys())]
    group_labels = [groups[gid]['name'] for gid in sorted(groups.keys())]
    
    bp = ax1.boxplot(group_data, labels=group_labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title(titles['group_comparison'], fontsize=14, fontweight='bold')
    ax1.set_ylabel(titles['val_loss'], fontsize=12)
    ax1.set_xlabel(titles['group'], fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Weight Decay vs 验证损失(分组显示)
    ax2 = plt.subplot(2, 3, 2)
    colors = ['blue', 'red', 'green', 'orange']
    for i, gid in enumerate(sorted(groups.keys())):
        group_data = groups[gid]['data']
        ax2.scatter(group_data['weight_decay'], group_data['best_val_loss'], 
                   c=colors[i], label=groups[gid]['name'], s=60, alpha=0.7)
    
    ax2.set_title(titles['wd_vs_loss'], fontsize=14, fontweight='bold')
    ax2.set_xlabel(titles['weight_decay'], fontsize=12)
    ax2.set_ylabel(titles['val_loss'], fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. KL Weight vs 验证损失(分组显示)
    ax3 = plt.subplot(2, 3, 3)
    for i, gid in enumerate(sorted(groups.keys())):
        group_data = groups[gid]['data']
        ax3.scatter(group_data['kl_weight'], group_data['best_val_loss'], 
                   c=colors[i], label=groups[gid]['name'], s=60, alpha=0.7)
    
    ax3.set_title(titles['kl_vs_loss'], fontsize=14, fontweight='bold')
    ax3.set_xlabel(titles['kl_weight'], fontsize=12)
    ax3.set_ylabel(titles['val_loss'], fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 各组最佳参数对比
    ax4 = plt.subplot(2, 3, 4)
    group_names = [groups[gid]['name'] for gid in sorted(groups.keys())]
    best_losses = [groups[gid]['best_trial']['best_val_loss'] for gid in sorted(groups.keys())]
    
    bars = ax4.bar(group_names, best_losses, color=colors[:len(group_names)], alpha=0.7)
    ax4.set_title(titles['best_params'], fontsize=14, fontweight='bold')
    ax4.set_ylabel(titles['val_loss'], fontsize=12)
    ax4.set_xlabel(titles['group'], fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 5. 相关性分析(每组内部)
    ax5 = plt.subplot(2, 3, 5)
    correlation_data = []
    for gid in sorted(groups.keys()):
        corr_wd = groups[gid]['correlations']['weight_decay']
        corr_kl = groups[gid]['correlations']['kl_weight']
        correlation_data.append([corr_wd, corr_kl])
    
    correlation_matrix = np.array(correlation_data).T
    im = ax5.imshow(correlation_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    
    ax5.set_xticks(range(len(group_names)))
    ax5.set_xticklabels(group_names, rotation=45)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Weight Decay', 'KL Weight'])
    ax5.set_title(titles['correlation'], fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i in range(len(group_names)):
        for j in range(2):
            text = ax5.text(i, j, f'{correlation_matrix[j, i]:.3f}',
                           ha="center", va="center", color="black" if abs(correlation_matrix[j, i]) < 0.5 else "white")
    
    plt.colorbar(im, ax=ax5, label='相关系数')
    
    # 6. 组内排名明细
    ax6 = plt.subplot(2, 3, 6)
    y_pos = 0
    y_labels = []
    y_positions = []
    all_bars = []
    
    for i, gid in enumerate(sorted(groups.keys())):
        group_data = groups[gid]['data'].sort_values('best_val_loss')
        group_name = groups[gid]['name']
        
        # 为每个组的试验创建条形
        for idx, (_, trial) in enumerate(group_data.iterrows()):
            bar = ax6.barh(y_pos, trial['best_val_loss'], 
                          color=colors[i], alpha=0.7, height=0.8)
            all_bars.extend(bar)
            y_labels.append(f"{group_name}\nT{trial['trial']}")
            y_positions.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # 组间间距
    
    ax6.set_yticks(y_positions)
    ax6.set_yticklabels(y_labels, fontsize=9)
    ax6.set_xlabel(titles['val_loss'], fontsize=12)
    ax6.set_title(titles['group_details'], fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

def print_analysis_summary(df):
    """打印分组分析摘要"""
    print("=" * 90)
    print("                       分组超参数优化分析报告")
    print("=" * 90)
    
    # 获取分组分析结果
    groups = analyze_groups(df)
    
    # 整体统计信息
    print(f"\n📊 整体统计信息:")
    print(f"   试验总数: {len(df)}")
    print(f"   分组数量: {len(groups)}")
    print(f"   整体最低验证损失: {df['best_val_loss'].min():.6f}")
    print(f"   整体最高验证损失: {df['best_val_loss'].max():.6f}")
    print(f"   整体平均验证损失: {df['best_val_loss'].mean():.6f}")
    
    # 各组详细分析
    print(f"\n📋 各组详细分析:")
    for gid in sorted(groups.keys()):
        group = groups[gid]
        print(f"\n  📊 {group['name']}:")
        print(f"     试验数量: {group['stats']['count']}")
        print(f"     最佳损失: {group['stats']['min_loss']:.6f}")
        print(f"     最差损失: {group['stats']['max_loss']:.6f}")
        print(f"     平均损失: {group['stats']['mean_loss']:.6f}")
        print(f"     标准差: {group['stats']['std_loss']:.6f}")
        
        # 最佳试验信息
        best = group['best_trial']
        print(f"     最佳试验: Trial {best['trial']}")
        print(f"       - Weight Decay: {best['weight_decay']:.6f}")
        print(f"       - KL Weight: {best['kl_weight']:.1f}")
        print(f"       - 验证损失: {best['best_val_loss']:.6f}")
        
        # 参数相关性
        print(f"     参数相关性(与损失):")
        print(f"       - Weight Decay: {group['correlations']['weight_decay']:.3f}")
        print(f"       - KL Weight: {group['correlations']['kl_weight']:.3f}")
    
    # 全局最佳试验
    best_trial = df.loc[df['best_val_loss'].idxmin()]
    best_group = None
    for gid in groups.keys():
        if best_trial['trial'] in groups[gid]['data']['trial'].values:
            best_group = groups[gid]['name']
            break
    
    print(f"\n🏆 全局最佳试验:")
    print(f"   试验编号: Trial {best_trial['trial']}")
    print(f"   所属组别: {best_group}")
    print(f"   验证损失: {best_trial['best_val_loss']:.6f}")
    print(f"   Weight Decay: {best_trial['weight_decay']:.6f}")
    print(f"   KL Weight: {best_trial['kl_weight']:.1f}")
    print(f"   Dropout: {best_trial['dropout']:.3f}")
    print(f"   观测数量: {best_trial['num_obs']:.0f}")
    print(f"   查询数量: {best_trial['num_queries']:.0f}")
    
    # 各组最佳参数排行
    print(f"\n🏅 各组最佳参数排行(按损失排序):")
    group_rankings = [(gid, groups[gid]['best_trial']['best_val_loss'], groups[gid]['name']) 
                     for gid in groups.keys()]
    group_rankings.sort(key=lambda x: x[1])
    
    for rank, (gid, loss, name) in enumerate(group_rankings, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        best = groups[gid]['best_trial']
        print(f"   {medal} {name}: {loss:.6f}")
        print(f"      Weight Decay: {best['weight_decay']:.6f}, KL Weight: {best['kl_weight']:.1f}")
    
    # 参数优化建议
    print(f"\n💡 参数优化建议:")
    print(f"   基于各组最佳结果的分析:")
    
    # 分析weight_decay范围
    best_wd_values = [groups[gid]['best_trial']['weight_decay'] for gid in groups.keys()]
    print(f"   Weight Decay 最优范围: {min(best_wd_values):.6f} - {max(best_wd_values):.6f}")
    print(f"   Weight Decay 平均最优值: {sum(best_wd_values)/len(best_wd_values):.6f}")
    
    # 分析kl_weight范围
    best_kl_values = [groups[gid]['best_trial']['kl_weight'] for gid in groups.keys()]
    print(f"   KL Weight 最优范围: {min(best_kl_values):.1f} - {max(best_kl_values):.1f}")
    print(f"   KL Weight 平均最优值: {sum(best_kl_values)/len(best_kl_values):.1f}")
    
    # 数据量影响分析
    print(f"\n🔍 数据量影响分析:")
    num_obs_losses = [(groups[gid]['data']['num_obs'].iloc[0], groups[gid]['stats']['min_loss']) 
                     for gid in groups.keys()]
    num_obs_losses.sort(key=lambda x: x[1])
    print(f"   最优表现的数据量组合: {num_obs_losses[0][0]}个观测")
    print(f"   对应最低损失: {num_obs_losses[0][1]:.6f}")
    
    print("\n" + "=" * 90)

def save_detailed_table(df):
    """保存详细的试验数据表格"""
    # 重新排序列以便更好的可读性
    available_columns = ['trial_name', 'trial', 'group', 'best_val_loss', 'weight_decay', 'kl_weight', 
                        'dropout', 'num_obs', 'num_queries', 'num_epochs', 'gpu_id']
    # 只选择存在的列
    columns_order = [col for col in available_columns if col in df.columns]
    
    df_display = df[columns_order].copy()
    
    # 格式化数值
    if 'best_val_loss' in df_display.columns:
        df_display['best_val_loss'] = df_display['best_val_loss'].apply(lambda x: f"{x:.6f}")
    if 'weight_decay' in df_display.columns:
        df_display['weight_decay'] = df_display['weight_decay'].apply(lambda x: f"{x:.6f}")
    if 'kl_weight' in df_display.columns:
        df_display['kl_weight'] = df_display['kl_weight'].apply(lambda x: f"{x:.1f}")
    if 'dropout' in df_display.columns:
        df_display['dropout'] = df_display['dropout'].apply(lambda x: f"{x:.3f}")
    if 'num_obs' in df_display.columns:
        df_display['num_obs'] = df_display['num_obs'].apply(lambda x: f"{x:.0f}")
    if 'num_queries' in df_display.columns:
        df_display['num_queries'] = df_display['num_queries'].apply(lambda x: f"{x:.0f}")
    if 'num_epochs' in df_display.columns:
        df_display['num_epochs'] = df_display['num_epochs'].apply(lambda x: f"{x:.0f}")
    
    # 重命名列
    column_mapping = {
        'trial_name': '试验名称',
        'trial': '试验编号',
        'group': '组别',
        'best_val_loss': '验证损失',
        'weight_decay': 'Weight Decay',
        'kl_weight': 'KL Weight',
        'dropout': 'Dropout',
        'num_obs': '观测数量',
        'num_queries': '查询数量',
        'num_epochs': '训练轮数',
        'gpu_id': 'GPU ID'
    }
    
    # 只重命名存在的列
    new_column_names = [column_mapping.get(col, col) for col in df_display.columns]
    df_display.columns = new_column_names
    
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
