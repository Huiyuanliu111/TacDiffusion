import numpy as np
import random
from train import train
import json
from datetime import datetime
import os
import shutil
import sys
import multiprocessing
import threading
import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# 创建线程锁用于更新配置文件
config_lock = threading.Lock()


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
            'weight_decay': 5 * 10 ** random.uniform(-4, -3),
            'kl_weight': random.uniform(5, 15.0),
            'dropout': 0.13,
            'completed': False,
            'best_val_loss': None,
            'sample_ratio': 1,
            'num_epochs': 20
        }
        trials_config.append(trial_config)
    
    # 保存配置
    with open(config_file, 'w') as f:
        json.dump(trials_config, f, indent=4)
    
    return trials_config


def run_trial_on_gpu(trial_config, gpu_id, results_dir):
    """在指定GPU上运行单个试验"""
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
    
    print(f"\n[GPU {gpu_id}] 开始试验 {trial}")
    print(f"[GPU {gpu_id}] 超参数配置:")
    print(f"[GPU {gpu_id}] weight_decay: {weight_decay:.6f}")
    print(f"[GPU {gpu_id}] kl_weight: {kl_weight:.2f}")
    print(f"[GPU {gpu_id}] dropout: {dropout:.3f}")
    print(f"[GPU {gpu_id}] sample_ratio: {sample_ratio:.3f}")
    
    try:
        # 运行训练，指定GPU
        best_val_loss = train(
            weight_decay=weight_decay,
            kl_weight=kl_weight,
            dropout=dropout,
            sample_ratio=sample_ratio,
            checkpoint_dir=checkpoint_dir,
            num_epochs=num_epochs,
            gpu_id=gpu_id,  # 新增GPU参数
            wandb_name=f"trial_{trial}_gpu{gpu_id}"  # 为每个试验设置唯一的wandb名称
        )
        
        # 更新试验状态
        trial_config['completed'] = True
        trial_config['best_val_loss'] = float(best_val_loss)
        trial_config['gpu_id'] = gpu_id
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(trial_dir, f"results_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump(trial_config, f, indent=4)
            
        print(f"[GPU {gpu_id}] 试验 {trial} 完成，验证损失: {best_val_loss:.4f}")
        return trial_config
        
    except Exception as e:
        print(f"[GPU {gpu_id}] 试验 {trial} 失败，错误信息: {str(e)}")
        trial_config['completed'] = False
        trial_config['error'] = str(e)
        return trial_config


def run_trial_on_cpu(trial_config, results_dir):
    """在CPU上运行单个试验"""
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
    
    print(f"\n试验 {trial + 1}")
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
        
        print(f"试验 {trial} 完成，验证损失: {best_val_loss:.4f}")
        return trial_config
        
    except Exception as e:
        print(f"试验 {trial} 失败，错误信息: {str(e)}")
        trial_config['completed'] = False
        trial_config['error'] = str(e)
        return trial_config


def update_config(trials_config, result):
    """线程安全地更新配置文件"""
    with config_lock:
        # 更新试验配置
        for i, t in enumerate(trials_config):
            if t['trial'] == result['trial']:
                trials_config[i] = result
                break
        
        # 保存当前进度
        with open("hp_search_config.json", 'w') as f:
            json.dump(trials_config, f, indent=4)


def search_multi_gpu(n_trials, max_gpus):
    """使用多GPU并行进行超参数搜索"""
    
    # 检查可用GPU数量
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练")
        max_gpus = 0
    else:
        available_gpus = torch.cuda.device_count()
        max_gpus = min(max_gpus, available_gpus)
        print(f"检测到 {available_gpus} 个GPU，将使用 {max_gpus} 个GPU进行并行训练")
        
        # 打印GPU信息
        for i in range(max_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 创建必要的目录
    results_dir = "hp_search_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)    # 确保logs目录存在
    
    # 加载或创建试验配置
    trials_config = load_or_create_trials_config(n_trials)
    
    # 获取未完成的试验
    remaining_trials = [t for t in trials_config if not t['completed']]
    
    print(f"\n总共 {n_trials} 个试验，还剩 {len(remaining_trials)} 个未完成")
    
    if not remaining_trials:
        print("所有试验已完成！")
        return
    
        # 使用线程池进行多GPU并行训练
        completed_trials = []

    with ThreadPoolExecutor(max_workers=max_gpus) as executor:
        # 提交初始任务
        future_to_trial = {}
        for i, trial_config in enumerate(remaining_trials[:max_gpus]):
            gpu_id = i
            future = executor.submit(run_trial_on_gpu, trial_config, gpu_id, results_dir)
            future_to_trial[future] = (trial_config, gpu_id)
        
        # 处理剩余任务
        remaining_task_idx = max_gpus
        
        # 等待任务完成并分配新任务
        for future in as_completed(future_to_trial):
            trial_config, gpu_id = future_to_trial[future]
            
            try:
                result = future.result()
                completed_trials.append(result)
                update_config(trials_config, result)
                
                print(f"\n[GPU {gpu_id}] 试验 {result['trial']} 完成")
                if result['completed']:
                    print(f"[GPU {gpu_id}] 验证损失: {result['best_val_loss']:.4f}")
                else:
                    print(f"[GPU {gpu_id}] 试验失败")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] 执行试验时出错: {str(e)}")
            
            # 如果还有未完成的任务，分配给这个GPU
            if remaining_task_idx < len(remaining_trials):
                next_trial = remaining_trials[remaining_task_idx]
                print(f"\n[GPU {gpu_id}] 开始新的试验 {next_trial['trial']}")
                
                future = executor.submit(run_trial_on_gpu, next_trial, gpu_id, results_dir)
                future_to_trial[future] = (next_trial, gpu_id)
                remaining_task_idx += 1

    # 找到最佳配置
    completed_trials_list = [t for t in trials_config if t['completed']]
    if completed_trials_list:
        best_trial = min(completed_trials_list, key=lambda x: x['best_val_loss'])
        print("\n" + "="*50)
        print("超参数搜索完成！")
        print("="*50)
        print("最佳超参数配置:")
        print(f"  试验编号: {best_trial['trial']}")
        print(f"  weight_decay: {best_trial['weight_decay']:.6f}")
        print(f"  kl_weight: {best_trial['kl_weight']:.2f}")
        print(f"  dropout: {best_trial['dropout']:.3f}")
        print(f"  最佳验证损失: {best_trial['best_val_loss']:.4f}")
        if max_gpus > 0:
            print(f"  训练GPU: {best_trial.get('gpu_id', 'N/A')}")
        print(f"  最佳模型保存在: {os.path.join(results_dir, 'trial_' + str(best_trial['trial']))}")
        
        # 显示所有完成试验的排名
        sorted_trials = sorted(completed_trials_list, key=lambda x: x['best_val_loss'])
        print("\n所有试验结果排名:")
        for i, trial in enumerate(sorted_trials):
            gpu_info = f" - GPU{trial.get('gpu_id', 'CPU')}" if max_gpus > 0 else ""
            print(f"  #{i+1}: 试验{trial['trial']} - 损失{trial['best_val_loss']:.4f}{gpu_info}")
    else:
        print("\n没有完成的试验")


if __name__ == "__main__":
    # 使用多GPU并行训练，默认使用7个GPU进行7个试验
    search_multi_gpu(n_trials=5, max_gpus=5)