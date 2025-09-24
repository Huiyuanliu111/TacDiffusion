#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照预定义的8组参数寻找每组的最佳模型
直接扫描output目录中的所有best_config.json文件
"""

import json
import pandas as pd
from pathlib import Path


def find_best_models_by_groups():
    """按照8组参数寻找最佳模型"""
    
    # 定义8组参数
    groups = [
        (1, 'Q5/O50', 5, 50),
        (2, 'Q5/O200', 5, 200), 
        (3, 'Q20/O50', 20, 50),
        (4, 'Q20/O400', 20, 400),
        (5, 'Q10/O100', 10, 100),
        (6, 'Q20/O200', 20, 200),
        (7, 'Q30/O300', 30, 300),
        (8, 'Q40/O400', 40, 400)
    ]

    print('🔍 按照8组参数寻找最佳模型...')
    print('=' * 80)

    output_dir = Path('output')
    best_models_report = []

    for group_id, group_name, target_queries, target_obs in groups:
        print(f'\n🎯 组{group_id}: {group_name}')
        
        # 寻找匹配的模型
        matching_models = []
        
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir():
                config_file = model_dir / 'best_config.json'
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        if (config.get('num_queries') == target_queries and 
                            config.get('num_obs') == target_obs):
                            matching_models.append({
                                'dir': model_dir,
                                'config': config,
                                'val_loss': config.get('best_val_loss', float('inf'))
                            })
                    except Exception as e:
                        continue
        
        if matching_models:
            # 找到验证损失最小的模型
            best_model = min(matching_models, key=lambda x: x['val_loss'])
            config = best_model['config']
            model_dir = best_model['dir']
            
            print(f'   ✅ 找到 {len(matching_models)} 个匹配模型')
            print(f'   🏆 最佳模型: {model_dir.name}')
            print(f'   📊 验证损失: {config.get("best_val_loss", "N/A"):.6f}')
            print(f'   🔧 超参数:')
            print(f'      - Weight Decay: {config.get("weight_decay", "N/A"):.6f}')
            print(f'      - KL Weight: {config.get("kl_weight", "N/A")}')
            print(f'      - Dropout: {config.get("dropout", "N/A"):.3f}')
            
            # 查找模型文件
            best_files = list(model_dir.glob('best_*.pth'))
            if best_files:
                best_file = best_files[0]
                size_mb = best_file.stat().st_size / (1024 * 1024)
                print(f'   📁 模型文件: {best_file.name} ({size_mb:.2f} MB)')
            else:
                print(f'   ❌ 未找到best_*.pth文件')
            
            # 添加到报告
            best_models_report.append({
                'group': f'组{group_id}: {group_name}',
                'group_id': group_id,
                'num_queries': target_queries,
                'num_obs': target_obs,
                'model_dir': str(model_dir),
                'model_name': model_dir.name,
                'val_loss': config.get('best_val_loss', 'N/A'),
                'weight_decay': config.get('weight_decay', 'N/A'),
                'kl_weight': config.get('kl_weight', 'N/A'),
                'dropout': config.get('dropout', 'N/A'),
                'model_file': str(best_files[0]) if best_files else 'N/A'
            })
        else:
            print(f'   ❌ 未找到匹配的模型 (Q{target_queries}/O{target_obs})')

    print(f'\n📋 总结:')
    print(f'找到 {len(best_models_report)}/8 组的最佳模型')

    # 保存详细报告
    if best_models_report:
        df = pd.DataFrame(best_models_report)
        report_file = 'best_models_by_8_groups.csv'
        df.to_csv(report_file, index=False, encoding='utf-8-sig')
        print(f'💾 报告已保存到: {report_file}')
        
        # 显示全局最佳模型
        if best_models_report:
            global_best = min(best_models_report, key=lambda x: x['val_loss'] if isinstance(x['val_loss'], (int, float)) else float('inf'))
            print(f'\n🏆 全局最佳模型:')
            print(f'   组别: {global_best["group"]}')
            print(f'   验证损失: {global_best["val_loss"]:.6f}')
            print(f'   模型目录: {global_best["model_dir"]}')
        
        return best_models_report
    else:
        print('❌ 未找到任何匹配的模型')
        return []


if __name__ == "__main__":
    find_best_models_by_groups()
