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

from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from helper_functions.data_split import RobotCustomDataset
from  act.policy import ACTPolicy

# Checkpoint functions
def save_checkpoint(model, optimizer, epoch, best_val_loss, global_step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'global_step': global_step
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
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        return start_epoch, best_val_loss, global_step
    else:
        print("No checkpoint found, starting from scratch")
        return 0, float('inf'), 1
    
def get_args_override():
    return {
        'ckpt_dir': 'checkpoints',
        'policy_class': 'ACT',
        'task_name': 'tactile',
        'seed': 42,
        'num_epochs': 300,
        'lr': 5e-5,
        'hidden_dim': 512,
        'kl_weight': 30.0,
        'num_queries': 200,
        'dropout': 0.1,
    }

def main():
    # Set paths and hyperparameters
    DATASET_PATH = "dataset"
    SAVE_DATA_DIR = "output" 
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)

    LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(LOG_DIR, exist_ok=True)

    n_epoch = 300 
    lrate = 5e-5 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_hidden = 512 
    batch_size = 64
   
    train_prop = 0.80

    sample_ratio = 0.001  

    args_override = get_args_override()
    
    model = ACTPolicy(args_override)
    print(f"num_queries: {model.model.num_queries}")

    Model_save_name = "ACT.pth"
    state_dataset = 'sensor_all.pkl'
    action_dataset = 'action_FF_all.pkl'

    tf = transforms.Compose([])

    num_workers = 8

    torch_data_train = RobotCustomDataset(
        DATASET_PATH, transform=tf, data_usage="train", train_prop=train_prop,
        state_dataset=state_dataset, action_dataset=action_dataset,
        num_queries=args_override['num_queries'], sample_ratio=sample_ratio
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    torch_data_val = RobotCustomDataset(
        DATASET_PATH, transform=tf, data_usage="valid", train_prop=train_prop,
        state_dataset=state_dataset, action_dataset=action_dataset, sample_ratio=sample_ratio,
        num_queries=args_override['num_queries']
    )
    dataload_val = DataLoader(
        torch_data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lrate)

    CHECKPOINT_DIR = "checkpoints"
    start_epoch, best_val_loss, global_step = load_checkpoint(model, optim, CHECKPOINT_DIR)

    writer = SummaryWriter(log_dir=LOG_DIR)

    for ep in tqdm(range(start_epoch, n_epoch), desc="Epoch"):
        model.train()
        optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

        pbar = tqdm(dataload_train)
        for x_batch, y_batch, is_pad in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(device)
            y_batch = y_batch.type(torch.FloatTensor).to(device)
            is_pad = is_pad.type(torch.BoolTensor).to(device)
            dummy_image = torch.zeros((x_batch.shape[0], 3, 224, 224)).to(device)
            
            loss_dict = model(qpos=x_batch, image=dummy_image, actions=y_batch, is_pad=is_pad)
            loss = loss_dict['loss']
            optim.zero_grad()
            loss.backward()
            pbar.set_description(f"train loss: {loss.detach().item():.4f}")
            writer.add_scalar('training_loss', loss.detach().item(), global_step)
            global_step += 1
            optim.step()
        if ep % 5 == 0:
            model.eval()
            loss_val, n_batch_val = 0, 0
            with torch.no_grad():
                for x_batch_val, y_batch_val, is_pad_val in tqdm(dataload_val, desc="Validation_Loss"):
                    x_batch_val = x_batch_val.type(torch.FloatTensor).to(device)
                    y_batch_val = y_batch_val.type(torch.FloatTensor).to(device)
                    is_pad_val = is_pad_val.type(torch.BoolTensor).to(device)
                    dummy_image = torch.zeros((x_batch_val.shape[0], 3, 224, 224)).to(device)
                    
                    loss_dict = model(qpos=x_batch_val, image=dummy_image, actions=y_batch_val, is_pad=is_pad_val)
                    loss_val_inner = loss_dict['loss']
                    loss_val += loss_val_inner.detach().item()
                    n_batch_val += 1

                avg_loss_val = loss_val / n_batch_val
                writer.add_scalar('validation_loss', avg_loss_val, global_step)
                tqdm.write(f"Epoch {ep+1}, validation loss: {avg_loss_val:.4f}")
            
            
            if avg_loss_val < best_val_loss:
                best_val_loss = avg_loss_val
                best_model_path = os.path.join(SAVE_DATA_DIR, f"best_{Model_save_name}")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

            save_checkpoint(model, optim, ep, best_val_loss, global_step, CHECKPOINT_DIR)

    writer.close()
    torch.save(model.state_dict(), os.path.join(SAVE_DATA_DIR, Model_save_name))

if __name__ == '__main__':
    multiprocessing.freeze_support()  
     # 添加命令行参数
    sys.argv.extend(['--ckpt_dir', 'checkpoints',
                    '--policy_class', 'ACT',

                    '--task_name', 'tactile',
                    '--seed', '42',
                    '--num_epochs', '300'])
    main()
