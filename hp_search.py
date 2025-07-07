import numpy as np
import random
from train import train
import json
from datetime import datetime
import os
import shutil
import sys

def random_search(n_trials=20):
    # 创建结果保存目录
    results_dir = "hp_search_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    for trial in range(n_trials):
        # 随机采样超参数
        weight_decay = 10 ** random.uniform(-5, -3)  # 1e-5 到 1e-3
        kl_weight = random.randint(1, 10)  # 1 到 10的整数
        dropout = random.uniform(0.05, 0.2)  # 0.05 到 0.2
        
        # 为每次试验创建独立的checkpoint目录
        checkpoint_dir = f"checkpoints_trial_{trial}"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)  # 删除之前的checkpoint
        os.makedirs(checkpoint_dir)
        
        print(f"\n试验 {trial + 1}/{n_trials}")
        print(f"超参数配置:")
        print(f"weight_decay: {weight_decay:.6f}")
        print(f"kl_weight: {kl_weight}")
        print(f"dropout: {dropout:.3f}")
        
        try:

            
            # 运行训练
            best_val_loss = train(
                weight_decay=weight_decay,
                kl_weight=kl_weight,
                dropout=dropout,
                checkpoint_dir=checkpoint_dir
            )
            
            # 记录结果
            trial_result = {
                'trial': trial,
                'weight_decay': weight_decay,
                'kl_weight': kl_weight,
                'dropout': dropout,
                'best_val_loss': float(best_val_loss)
            }
            results.append(trial_result)
            
            # 保存当前结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(results_dir, f"hp_search_results_{timestamp}.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=4)
                
            print(f"验证损失: {best_val_loss:.4f}")
            
            # 清理checkpoint目录（可选，取决于是否需要保留）
            shutil.rmtree(checkpoint_dir)
            
        except Exception as e:
            print(f"试验失败，错误信息: {str(e)}")
            # 清理失败试验的checkpoint目录
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            continue
    
    # 找到最佳配置
    best_trial = min(results, key=lambda x: x['best_val_loss'])
    print("\n最佳超参数配置:")
    print(f"weight_decay: {best_trial['weight_decay']:.6f}")
    print(f"kl_weight: {best_trial['kl_weight']}")
    print(f"dropout: {best_trial['dropout']:.3f}")
    print(f"最佳验证损失: {best_trial['best_val_loss']:.4f}")

if __name__ == "__main__":
    random_search() 