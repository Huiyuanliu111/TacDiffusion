import numpy as np
import random
from train import train
import json
from datetime import datetime
import os
import shutil
import sys

def load_or_create_trials_config(n_trials):
    """加载或创建试验配置"""
    config_file = "hp_search_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # 创建新的试验配置
    trials_config = []
    for trial in range(n_trials):
        trial_config = {
            'trial': trial,
            'weight_decay': 10 ** random.uniform(-4, -3),
            'kl_weight': random.uniform(1, 10.0),
            'dropout': random.uniform(0.1, 0.2),
            'completed': False,
            'best_val_loss': None,
            'sample_ratio': 0.5,
            'num_epochs': 10
        }
        trials_config.append(trial_config)
    
    # 保存配置
    with open(config_file, 'w') as f:
        json.dump(trials_config, f, indent=4)
    
    return trials_config

def search(n_trials=4):
    # 创建必要的目录
    results_dir = "hp_search_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)    # 确保logs目录存在
    
    # 加载或创建试验配置
    trials_config = load_or_create_trials_config(n_trials)
    
    # 获取未完成的试验
    remaining_trials = [t for t in trials_config if not t['completed']]
    
    print(f"总共 {n_trials} 个试验，还剩 {len(remaining_trials)} 个未完成")
    
    for trial_config in remaining_trials:
        trial = trial_config['trial']
        weight_decay = trial_config['weight_decay']
        kl_weight = trial_config['kl_weight']
        dropout = trial_config['dropout']
        sample_ratio = trial_config['sample_ratio']
        num_epochs = trial_config['num_epochs']

        # 为每次试验创建独立的目录结构
        trial_dir = os.path.join(results_dir, f"trial_{trial}")
        checkpoint_dir = os.path.join(trial_dir, "checkpoints")
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n试验 {trial + 1}/{n_trials}")
        print(f"超参数配置:")
        print(f"weight_decay: {weight_decay:.6f}")
        print(f"kl_weight: {kl_weight:.2f}")
        print(f"dropout: {dropout:.3f}")
        print(f"sample_ratio: {sample_ratio:.3f}")
        try:
            # 为每个trial创建唯一的wandb名称
            wandb_name = f"trial_{trial}_wd{weight_decay:.1e}_kl{kl_weight:.1f}_dp{dropout:.3f}"
            
            # 运行训练
            best_val_loss = train(
                weight_decay=weight_decay,
                kl_weight=kl_weight,
                dropout=dropout,
                sample_ratio=sample_ratio,
                checkpoint_dir=checkpoint_dir,
                num_epochs=num_epochs,
                trial_id=trial,
                wandb_name=wandb_name
            )
            
            # 更新试验状态
            trial_config['completed'] = True
            trial_config['best_val_loss'] = float(best_val_loss)
            
            # 保存当前进度
            with open("hp_search_config.json", 'w') as f:
                json.dump(trials_config, f, indent=4)
            
            # 保存详细结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(trial_dir, f"results_{timestamp}.json")
            with open(result_file, 'w') as f:
                json.dump(trial_config, f, indent=4)
                
            print(f"验证损失: {best_val_loss:.4f}")
            
            # 清理日志
            for log_dir in os.listdir("logs/fit"):
                shutil.rmtree(os.path.join("logs/fit", log_dir))  # 清理本次trial的日志
            
        except Exception as e:
            print(f"试验失败，错误信息: {str(e)}")
            continue
    
    # 找到最佳配置
    completed_trials = [t for t in trials_config if t['completed']]
    if completed_trials:
        best_trial = min(completed_trials, key=lambda x: x['best_val_loss'])
        print("\n最佳超参数配置:")
        print(f"weight_decay: {best_trial['weight_decay']:.6f}")
        print(f"kl_weight: {best_trial['kl_weight']:.2f}")
        print(f"dropout: {best_trial['dropout']:.3f}")
        print(f"最佳验证损失: {best_trial['best_val_loss']:.4f}")
        print(f"最佳模型保存在: {os.path.join(results_dir, 'trial_' + str(best_trial['trial']))}")
    else:
        print("\n没有完成的试验")

if __name__ == "__main__":
    search() 