import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
import multiprocessing
import platform
import json
import wandb
import wandb

from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from helper_functions.data_split import RobotCustomDataset
from  act.policy import ACTPolicy

# Checkpoint functions
def save_checkpoint(model, optimizer, epoch, best_val_loss, global_step, checkpoint_dir, patience_counter):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'global_step': global_step,
        'patience_counter': patience_counter
    }
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        global_step = checkpoint['global_step']
        patience_counter = checkpoint.get('patience_counter', 0)  # 兼容旧的checkpoint
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        return start_epoch, best_val_loss, global_step, patience_counter
    else:
        print("No checkpoint found, starting from scratch")
        return 0, float('inf'), 1, 0
    

def train(weight_decay=1e-4, kl_weight=1, dropout=0.1, sample_ratio=0.5, num_epochs=20, checkpoint_dir="checkpoints", trial_id=None, use_wandb=True, wandb_project="act_tac", wandb_name=None, gpu_id=None):
    # 如果提供了trial_id，则更新hp_search_config
    hp_config_path = "hp_search_config.json"
    if trial_id is not None and os.path.exists(hp_config_path):
        with open(hp_config_path, 'r') as f:
            hp_config = json.load(f)
    
    # Set paths and hyperparameters
    DATASET_PATH = "dataset"

    # 为每组超参数创建独立的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gpu_suffix = f"_gpu{gpu_id}" if gpu_id is not None else ""
    run_name = f"wd{weight_decay:.1e}_kl{kl_weight:.1f}_dp{dropout:.3f}_{timestamp}{gpu_suffix}"
    
    SAVE_DATA_DIR = os.path.join("output", run_name)
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)

    LOG_DIR = os.path.join("logs/fit", run_name)
    os.makedirs(LOG_DIR, exist_ok=True)

    n_epoch = num_epochs
    lrate = 5e-5 
    
    # 设置GPU设备
    if gpu_id is not None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"警告: GPU {gpu_id} 不可用，使用CPU")
            device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    n_hidden = 256
    batch_size = 64 # 进一步减小batch_size以避免CUDA内存不足
    
    num_workers = 16
    train_prop = 0.80
    sample_ratio = 1  
    sample_ratio = 1  
    num_queries = 200
    num_obs = 500
    patience = 3


    args_override = {
        'num_epochs': n_epoch,
        'lr': lrate,
        'hidden_dim': n_hidden,
        'kl_weight': kl_weight,  # 使用传入的参数
        'num_queries': num_queries,
        'dropout': dropout,      # 使用传入的参数
        "weight_decay": weight_decay,  # 使用传入的参数
        'train_prop': train_prop,
        'batch_size': batch_size,
        'num_obs': num_obs,
        'sample_ratio': sample_ratio,
        'patience': patience
        'sample_ratio': sample_ratio,
        'patience': patience
    }

    model = ACTPolicy(args_override)
    
    # 初始化wandb
    if use_wandb:
        wandb_run_name = wandb_name
        wandb.init(
            project="act_tac",
            entity="huiyuan_tac",
            name=wandb_run_name,
            config=args_override,
            tags=["ACT", "diffusion", "tactile"]
        )
        # 监视模型
        wandb.watch(model, log="all", log_freq=100)
    print(f"num_queries: {model.model.num_queries}")
    
    # 在wandb中记录模型架构信息
    if use_wandb:
        wandb.config.update({
            "model_num_queries": model.model.num_queries,
            "model_type": "ACTPolicy"
        })
    
    # 在wandb中记录模型架构信息
    if use_wandb:
        wandb.config.update({
            "model_num_queries": model.model.num_queries,
            "model_type": "ACTPolicy"
        })

    # 为每次运行创建独特的模型保存名称
    Model_save_name = f"ACT_{run_name}.pth"
    state_dataset = 'sensor_all.pkl'
    action_dataset = 'action_FF_all.pkl'

    tf = transforms.Compose([])


    torch_data_train = RobotCustomDataset(
        DATASET_PATH, transform=tf, data_usage="train", train_prop= args_override['train_prop'],
        state_dataset=state_dataset, action_dataset=action_dataset,
        num_queries=args_override['num_queries'], sample_ratio=sample_ratio,
        num_obs=args_override['num_obs']
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=args_override['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    torch_data_val = RobotCustomDataset(
        DATASET_PATH, transform=tf, data_usage="valid", train_prop=args_override['train_prop'],
        state_dataset=state_dataset, action_dataset=action_dataset, sample_ratio=sample_ratio,
        num_queries=args_override['num_queries'], num_obs=args_override['num_obs']
    )
    dataload_val = DataLoader(
        torch_data_val, batch_size=args_override['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args_override['lr'], weight_decay=weight_decay)

    start_epoch, best_val_loss, global_step, patience_counter = load_checkpoint(model, optim, checkpoint_dir)

    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # 早停相关变量
    best_val_loss = float('inf') 

    for ep in tqdm(range(start_epoch, n_epoch), desc="Epoch"):
        model.train()
        optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

        pbar = tqdm(dataload_train)
        for x_batch, y_batch, state_pad, actions_pad in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(device)
            y_batch = y_batch.type(torch.FloatTensor).to(device)
            state_pad = state_pad.type(torch.BoolTensor).to(device)
            actions_pad = actions_pad.type(torch.BoolTensor).to(device)
            dummy_image = torch.zeros((x_batch.shape[0], 3, 224, 224)).to(device)
            
            loss_dict = model(qpos=x_batch, image=dummy_image, actions=y_batch, state_pad=state_pad, actions_pad=actions_pad)
            loss = loss_dict['loss']
            optim.zero_grad()
            loss.backward()
            pbar.set_description(f"train loss: {loss.detach().item():.4f}")
            writer.add_scalar('training_loss', loss.detach().item(), global_step)
            
            # 记录到wandb
            if use_wandb:
                wandb.log({
                    'train_loss': loss.detach().item(),
                    'learning_rate': optim.param_groups[0]["lr"],
                    'epoch': ep,
                    'global_step': global_step
                }, step=global_step)
            
            
            # 记录到wandb
            if use_wandb:
                wandb.log({
                    'train_loss': loss.detach().item(),
                    'learning_rate': optim.param_groups[0]["lr"],
                    'epoch': ep,
                    'global_step': global_step
                }, step=global_step)
            
            global_step += 1
            optim.step()

        save_checkpoint(model, optim, ep, best_val_loss, global_step, checkpoint_dir, patience_counter)

        if ep % 1  == 0:
            model.eval()
            loss_val, n_batch_val = 0, 0
            with torch.no_grad():
                for x_batch_val, y_batch_val, state_pad_val, actions_pad_val in tqdm(dataload_val, desc="Validation_Loss"):
                    x_batch_val = x_batch_val.type(torch.FloatTensor).to(device)
                    y_batch_val = y_batch_val.type(torch.FloatTensor).to(device)
                    state_pad_val = state_pad_val.type(torch.BoolTensor).to(device)
                    actions_pad_val = actions_pad_val.type(torch.BoolTensor).to(device)
                    dummy_image = torch.zeros((x_batch_val.shape[0], 3, 224, 224)).to(device)
                    
                    loss_dict = model(qpos=x_batch_val, image=dummy_image, actions=y_batch_val, 
                                    state_pad=state_pad_val, actions_pad=actions_pad_val)
                    loss_val_inner = loss_dict['loss']
                    loss_val += loss_val_inner.detach().item()
                    n_batch_val += 1

                avg_loss_val = loss_val / n_batch_val
                # 使用epoch数作为x轴，只记录每个epoch的平均验证损失
                writer.add_scalar('validation_loss', avg_loss_val, ep)
                
                # 记录到wandb
                if use_wandb:
                    wandb.log({
                        'val_loss': avg_loss_val,
                        'epoch': ep
                    }, step=global_step)
                
                # 记录到wandb
                if use_wandb:
                    wandb.log({
                        'val_loss': avg_loss_val,
                        'epoch': ep
                    }, step=global_step)

                tqdm.write(f"Epoch {ep+1}, validation loss: {avg_loss_val:.4f}")
            
            
            if avg_loss_val < best_val_loss:
                best_val_loss = avg_loss_val
                best_model_path = os.path.join(SAVE_DATA_DIR, f"best_{Model_save_name}")
                torch.save(model.state_dict(), best_model_path)
                print(f"新的最佳模型已保存 - 验证损失: {best_val_loss:.4f}")
                patience_counter = 0  # 重置patience计数器
                
                # 记录最佳模型到wandb
                if use_wandb:
                    wandb.log({
                        'best_val_loss': best_val_loss,
                        'best_model_epoch': ep
                    })
                    # 保存模型artifact
                    artifact = wandb.Artifact(f'best_model_{wandb.run.id}', type='model')
                    artifact.add_file(best_model_path)
                    wandb.log_artifact(artifact)
                
                # 记录最佳模型到wandb
                if use_wandb:
                    wandb.log({
                        'best_val_loss': best_val_loss,
                        'best_model_epoch': ep
                    })
                    # 保存模型artifact
                    artifact = wandb.Artifact(f'best_model_{wandb.run.id}', type='model')
                    artifact.add_file(best_model_path)
                    wandb.log_artifact(artifact)
                
                # 同时保存超参数配置
                config_path = os.path.join(SAVE_DATA_DIR, "best_config.json")
                config_data = args_override.copy()
                config_data['best_val_loss'] = best_val_loss
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=4)
            else:
                patience_counter += 1
                print(f"验证损失未改善，当前patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"Early stopping 触发！{patience}个epoch内验证损失未改善")
                break



    writer.close()
    final_model_path = os.path.join(SAVE_DATA_DIR, Model_save_name)
    torch.save(model.state_dict(), final_model_path)
    
    # 记录最终结果到wandb
    if use_wandb:
        wandb.log({
            'final_val_loss': best_val_loss,
            'training_completed': True
        })
        # 保存最终模型artifact
        final_artifact = wandb.Artifact(f'final_model_{wandb.run.id}', type='model')
        final_artifact.add_file(final_model_path)
        wandb.log_artifact(final_artifact)
        
        # 保存训练配置
        config_artifact = wandb.Artifact(f'config_{wandb.run.id}', type='config')
        config_artifact.add_file(os.path.join(SAVE_DATA_DIR, "best_config.json"))
        wandb.log_artifact(config_artifact)
        
        wandb.finish()
    
    
    # 记录最终结果到wandb
    if use_wandb:
        wandb.log({
            'final_val_loss': best_val_loss,
            'training_completed': True
        })
        # 保存最终模型artifact
        final_artifact = wandb.Artifact(f'final_model_{wandb.run.id}', type='model')
        final_artifact.add_file(final_model_path)
        wandb.log_artifact(final_artifact)
        
        # 保存训练配置
        config_artifact = wandb.Artifact(f'config_{wandb.run.id}', type='config')
        config_artifact.add_file(os.path.join(SAVE_DATA_DIR, "best_config.json"))
        wandb.log_artifact(config_artifact)
        
        wandb.finish()
    
    print(f"\n训练完成！")
    print(f"最终模型保存于: {final_model_path}")
    print(f"最佳验证损失: {best_val_loss:.4f}")

    # 如果是超参数搜索的一部分，更新配置文件
    if trial_id is not None and os.path.exists(hp_config_path):
        hp_config[trial_id]["best_val_loss"] = float(best_val_loss)
        with open(hp_config_path, 'w') as f:
            json.dump(hp_config, f, indent=4)
        
    return best_val_loss

if __name__ == '__main__':
    multiprocessing.freeze_support()

    train()