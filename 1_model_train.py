import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys

from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from helper_functions.data_split import RobotCustomDataset
from  act.policy import ACTPolicy

# Checkpoint functions
def save_checkpoint(model, optimizer, epoch, best_val_loss, global_step, checkpoint_dir):
    """保存checkpoint"""
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
    """加载checkpoint，返回起始epoch、最佳验证损失和global_step"""
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

# 添加命令行参数
sys.argv.extend(['--ckpt_dir', 'checkpoints',
                '--policy_class', 'ACT',
                '--task_name', 'tactile',
                '--seed', '42',
                '--num_epochs', '4000'])

# Set paths and hyperparameters
DATASET_PATH = "dataset"
SAVE_DATA_DIR = "output" 
os.makedirs(SAVE_DATA_DIR, exist_ok=True)

LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)

n_epoch = 5000 
lrate = 1e-5 
device = "cuda" if torch.cuda.is_available() else "cpu"
n_hidden = 512 
batch_size = 32 
n_T = 50
net_type = "fc"
drop_prob = 0.1
train_prop = 0.80
use_prev = True

args_override = {
    # 必需的命令行参数
    'ckpt_dir': 'checkpoints',
    'policy_class': 'ACT',
    'task_name': 'tactile',
    'seed': 42,
    'num_epochs': n_epoch,
    # 需要覆盖的模型参数
    'lr': lrate,
    'hidden_dim': n_hidden,
    'kl_weight': 10.0,
    'num_queries': 100
}
# Initialize the model and optimizer
model = ACTPolicy(args_override)

# 打印模型内置超参数

print(f"查询数量 (num_queries): {model.model.num_queries}")





Model_save_name = "ACT.pth"
state_dataset = 'sensor_all.pkl'
action_dataset = 'action_FF_all.pkl'
#state_dataset = 'robot_state_train.pkl'
#action_dataset = 'robot_action_train.pkl'

# Load training and validation data
tf = transforms.Compose([])

torch_data_train = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="train", train_prop=train_prop,
    state_dataset=state_dataset, action_dataset=action_dataset,
    num_queries=args_override['num_queries']
)
dataload_train = DataLoader(
    torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
)

torch_data_val = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="valid", train_prop=train_prop,
    state_dataset=state_dataset, action_dataset=action_dataset,
    num_queries=args_override['num_queries']
)
dataload_val = DataLoader(
    torch_data_val, batch_size=batch_size, shuffle=False, num_workers=0
)




model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lrate)

# Checkpoint setup
CHECKPOINT_DIR = "checkpoints"
start_epoch, best_val_loss, global_step = load_checkpoint(model, optim, CHECKPOINT_DIR)

# Set up TensorBoard logging
writer = SummaryWriter(log_dir=LOG_DIR)

# Main training loop
for ep in tqdm(range(start_epoch, n_epoch), desc="Epoch"):

    model.train()

    # Learning rate decay
    optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

    # Training loop
    pbar = tqdm(dataload_train)
    for x_batch, y_batch, is_pad in pbar:
        x_batch = x_batch.type(torch.FloatTensor).to(device)
        y_batch = y_batch.type(torch.FloatTensor).to(device)
        is_pad = is_pad.type(torch.BoolTensor).to(device)
        # 创建空的图像tensor
        dummy_image = torch.zeros((x_batch.shape[0], 3, 224, 224)).to(device)  # 标准图像大小
        
        loss_dict = model(qpos=x_batch, image=dummy_image, actions=y_batch, is_pad=is_pad)
        loss = loss_dict['loss']  # 获取总loss
        optim.zero_grad()
        loss.backward()
        pbar.set_description(f"train loss: {loss.detach().item():.4f}")
        # Log training loss to TensorBoard
        writer.add_scalar('training_loss', loss.detach().item(), global_step)
        global_step += 1

        optim.step()

    if (ep + 1) % 5 == 0:  # Validate every 5 epochs
        model.eval()
        loss_val, n_batch_val = 0, 0
        with torch.no_grad():
            for x_batch_val, y_batch_val, is_pad_val in tqdm(dataload_val, desc="Validation_Loss"):
                x_batch_val = x_batch_val.type(torch.FloatTensor).to(device)
                y_batch_val = y_batch_val.type(torch.FloatTensor).to(device)
                is_pad_val = is_pad_val.type(torch.BoolTensor).to(device)
                
                # 创建空的图像tensor
                dummy_image = torch.zeros((x_batch_val.shape[0], 3, 224, 224)).to(device)
                
                loss_dict = model(qpos=x_batch_val, image=dummy_image, actions=y_batch_val, is_pad=is_pad_val)
                loss_val_inner = loss_dict['loss']
                loss_val += loss_val_inner.detach().item()
                n_batch_val += 1

            avg_loss_val = loss_val / n_batch_val
            writer.add_scalar('validation_loss', avg_loss_val, global_step)
            tqdm.write(f"Epoch {ep+1}, validation loss: {avg_loss_val:.4f}")
            
            save_checkpoint(model, optim, ep, best_val_loss, global_step, CHECKPOINT_DIR)
            
            # Save best model
            if avg_loss_val < best_val_loss:
                best_val_loss = avg_loss_val
                best_model_path = os.path.join(SAVE_DATA_DIR, f"best_{Model_save_name}")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

# Close TensorBoard writer
writer.close()

# Save the trained model
torch.save(model.state_dict(), os.path.join(SAVE_DATA_DIR, Model_save_name))
