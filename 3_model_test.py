import os
import torch
import time
import xlsxwriter
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
from torchvision import transforms
from helper_functions.data_split import RobotCustomDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set paths and directories
DATASET_PATH = "dataset"
SAVE_FIGURE_DIR = "figures/ACT"
FIGURE_ACTION_DIR = os.path.join(SAVE_FIGURE_DIR, "figures_action")
FIGURE_state_DIR = os.path.join(SAVE_FIGURE_DIR, "figures_state")
FIGURE_ERROR_DIR = os.path.join(SAVE_FIGURE_DIR, "figures_error")
interval_length = 1000  # Reduced for testing - Length of valid data

temporal_agg = True  # 启用时间加权聚合
query_frequency = 1  # 查询频率
num_queries = 200  # 序列长度
k = 0.01  

# Set device: use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')

# Dataset files
state_dataset = 'sensor_all.pkl'
action_dataset = 'action_FF_all.pkl'
print(f'state_dataset: {state_dataset}')
print(f'action_dataset: {action_dataset}')

# Model file
model_name = 'output/ACT.onnx'
print(f'model_name: {model_name}')

# Load datasets
tf = transforms.Compose([])
torch_data_test = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="test", train_prop=0.01,
    state_dataset=state_dataset, action_dataset=action_dataset,
    normalize_data=True  # 🔥 启用数据标准化
)

# Extract input and output dimensions
x_shape = torch_data_test.state.shape[1]
y_dim = torch_data_test.action.shape[1]
y_true = torch_data_test.action

# 🔥 获取真实数据（反标准化）
if hasattr(torch_data_test, 'denormalize_actions'):
    y_true_original = torch_data_test.denormalize_actions(y_true)
    print("✅ 使用反标准化的真实数据进行评估")
else:
    y_true_original = y_true
    print("⚠️ 未找到反标准化方法，使用原始数据")

print(f'y_dim: {y_dim}')
print('data import success!')

# Enable graph optimization
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Create ONNX inference session
print("Loading ONNX model...")
ort_session = ort.InferenceSession(model_name, sess_options)
print("ONNX model loaded successfully!")

# Determine number of intervals in the dataset
dim_validation = torch_data_test.state.shape[0]
max_intervals = dim_validation // interval_length
num_intervals = min(3, max_intervals)  # Limit to 3 intervals for testing
print(f"Total validation samples: {dim_validation}")
print(f"Max possible intervals: {max_intervals}")
print(f"Number of intervals to process: {num_intervals} (limited for testing)")
print(f"Interval length: {interval_length}")

# Labels for predictions and ground truth
label_pred = ['f_x_pred', 'f_y_pred', 'f_z_pred', 'tau_x_pred', 'tau_y_pred', 'tau_z_pred']
label_true = ['f_x_true', 'f_y_true', 'f_z_true', 'tau_x_true', 'tau_y_true', 'tau_z_true']
data_label = ['f_x_ext', 'f_y_ext', 'f_z_ext', 
              'tau_x_ext', 'tau_y_ext', 'tau_z_ext', 
              'f_x_in', 'f_y_in', 'f_z_in', 
              'tau_x_in', 'tau_y_in', 'tau_z_in', 
              'v_x', 'v_y', 'v_z',
              'w_x', 'w_y', 'w_z']

subplot_label = ['wrench_ext_f',
                 'wrench_ext_tau',
                 'wrench_inner_f',
                 'wrench_inner_tau',
                 'linear velocity',
                 'angular velocity']

# Initialize DataFrame to store error metrics and inference speed
error_df = pd.DataFrame(columns=["Figure Name", "MAE", "RMSE", "Inference Speed (samples/second)"])

# Lists to store all ground truth, predictions, and inference speeds for overall error calculation
all_y_true = []
all_y_pred = []
all_inference_speeds = []

# Create directories to save figures if they do not exist
if not os.path.exists(SAVE_FIGURE_DIR):
    os.makedirs(SAVE_FIGURE_DIR)
if not os.path.exists(FIGURE_ACTION_DIR):
    os.makedirs(FIGURE_ACTION_DIR)
if not os.path.exists(FIGURE_state_DIR):
    os.makedirs(FIGURE_state_DIR)
if not os.path.exists(FIGURE_ERROR_DIR):
    os.makedirs(FIGURE_ERROR_DIR)

print("Starting inference loop...")
# Iterate through intervals and perform inference
for interval_idx in range(num_intervals):
    print(f"Processing interval {interval_idx + 1}/{num_intervals}")
    start_idx = interval_idx * interval_length
    end_idx = start_idx + interval_length
    idxs = range(start_idx, end_idx)
    y_pred = np.zeros((interval_length, 200, y_dim))  # Array to store prediction results with sequence dimension


    
    # 初始化时间动作矩阵
    all_time_actions = torch.zeros((interval_length, interval_length + num_queries - 1, y_dim))

    start_time = time.time()
    print(f"  Starting inference for {len(idxs)} samples...")

    # Perform inference on each sample in the interval
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            if i % 100 == 0:  # Print progress every 100 samples
                print(f"    Processing sample {i}/{len(idxs)}")
            x_eval = torch.Tensor(torch_data_test.state[idx]).type(torch.FloatTensor).to(device)
            x_eval_ = x_eval.repeat(1, 1).cpu().numpy()
            
            # 执行推理
            if i % query_frequency == 0:
                all_actions = ort_session.run(['output'], {'qpos': x_eval_})[0]
                all_actions = torch.from_numpy(all_actions[0])  # [sequence, action_dim]
            
            if temporal_agg:
                # 时间加权聚合
                end_time_idx = min(i + num_queries, all_time_actions.shape[1])
                seq_length = min(num_queries, end_time_idx - i)
                all_time_actions[i, i:i+seq_length] = all_actions[:seq_length]
                actions_for_curr_step = all_time_actions[:i+1, i]
                actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                if len(actions_for_curr_step) > 0:
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[0:1]  # 取第一个动作
            else:
                raw_action = all_actions[i % query_frequency:i % query_frequency + 1]
            
            # 存储结果
            y_pred[i] = all_actions.numpy()  # 保存完整序列用于可视化
            
            # 如果需要单步动作，可以使用 raw_action
            # y_pred_single[i] = raw_action.numpy()

    end_time = time.time()
    inference_time = end_time - start_time
    inference_speeds = interval_length / inference_time
    print(f"Interval {interval_idx}: inference speeds: {inference_speeds} sample/second")

    # Calculate errors for the current interval
    y_true_interval = y_true_original[start_idx:end_idx]  # 🔥 使用反标准化的真实数据
    # For error calculation, use only the first action from the sequence
    y_pred_first_action = y_pred[:, 0, :]  # Take first action from sequence: (interval_length, y_dim)
    
    # 🔥 反标准化预测结果
    if hasattr(torch_data_test, 'denormalize_actions'):
        y_pred_first_action_original = torch_data_test.denormalize_actions(y_pred_first_action)
        print(f"✅ 区间 {interval_idx + 1}: 预测结果已反标准化")
        # 在反标准化之前打印数据统计信息
        print(f"标准化数据统计: mean={np.mean(y_pred_first_action):.4f}, std={np.std(y_pred_first_action):.4f}")
        print(f"使用的均值: {torch_data_test.action_mean}")
        print(f"使用的标准差: {torch_data_test.action_std}")
        # 反标准化后打印统计信息
        denormalized = y_pred_first_action * torch_data_test.action_std + torch_data_test.action_mean
        print(f"反标准化数据统计: mean={np.mean(denormalized):.4f}, std={np.std(denormalized):.4f}")
    else:
        y_pred_first_action_original = y_pred_first_action
        print(f"⚠️ 区间 {interval_idx + 1}: 未找到反标准化方法")
    
    # 使用原始尺度数据计算误差
    mae = mean_absolute_error(y_true_interval, y_pred_first_action_original)
    rmse = np.sqrt(mean_squared_error(y_true_interval, y_pred_first_action_original))
    
    print(f"区间 {interval_idx + 1}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # Add the current interval's error and inference speed to the DataFrame
    figure_saved_name = f"action_data_{start_idx}_{end_idx}.png"
    new_row = pd.DataFrame({
        "Figure Name": [figure_saved_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "Inference Speed (samples/second)": [inference_speeds]
    })
    if error_df.empty:
        error_df = new_row
    else:
        error_df = pd.concat([error_df, new_row], ignore_index=True)

    # Add to the list for overall error calculation
    all_y_true.append(y_true_interval)
    all_y_pred.append(y_pred_first_action_original)  # 🔥 使用反标准化的预测结果
    all_inference_speeds.append(inference_speeds)

    # Plot and save the action data figures
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    for i in range(6):
        row, col = divmod(i, 2)
        # 🔥 使用反标准化的数据进行可视化
        axs[row, col].plot(y_pred_first_action_original[:, i], label=label_pred[i], linestyle='-', color='blue')
        axs[row, col].plot(y_true_interval[:, i], label=label_true[i], linestyle='--', color='orange')

        axs[row, col].set_xlabel('Index')
        axs[row, col].set_ylabel('Value')
        axs[row, col].set_title(f'Line Plot of {label_pred[i]} and {label_true[i]}')
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_ACTION_DIR, figure_saved_name))
    plt.close(fig)
    print(f"Saved figure {figure_saved_name} in {FIGURE_ACTION_DIR}")

    # Plot and save the state data figures
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    for k in range(6):
        row, col = divmod(k, 2)  # Calculate row and column index
        start_col = k * 3

        # Plot up to three columns of state data in the current subplot
        for l in range(3):
            col_index = start_col + l
            if col_index < 18:  # Ensure col_index is within bounds
                axs[row, col].plot(torch_data_test.state[start_idx:end_idx, col_index], label=f'{data_label[col_index]}', linestyle='-')

        axs[row, col].set_xlabel('Index')
        axs[row, col].set_ylabel('Value')
        axs[row, col].set_title(f'{subplot_label[k]}')
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()  # Adjust subplots to fit into figure area

    state_figure_saved_name = f"state_data_{start_idx}_{end_idx}.png"
    plt.savefig(os.path.join(FIGURE_state_DIR, state_figure_saved_name))
    plt.close(fig)
    print(f"Saved state figure {state_figure_saved_name} in {FIGURE_state_DIR}")
    
    # Generate intermediate error plot for this interval
    if len(error_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        error_df.plot(kind='bar', x='Figure Name', y=['MAE', 'RMSE'], ax=ax)
        plt.xticks(rotation=45)
        plt.title(f'MAE and RMSE up to interval {interval_idx + 1}')
        plt.xlabel('Figure Name')
        plt.ylabel('Error')
        plt.tight_layout()
        intermediate_error_plot = f"Intermediate_Error_Plot_Interval_{interval_idx + 1}.png"
        plt.savefig(os.path.join(FIGURE_ERROR_DIR, intermediate_error_plot))
        plt.close(fig)
        print(f"Saved intermediate error plot {intermediate_error_plot} in {FIGURE_ERROR_DIR}")

# Calculate overall error metrics
all_y_true = np.vstack(all_y_true)
all_y_pred = np.vstack(all_y_pred)
overall_mae = mean_absolute_error(all_y_true, all_y_pred)
overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
overall_avg_speed = np.mean(all_inference_speeds)

# Add overall error metrics and average inference speed to the DataFrame
overall_row = pd.DataFrame({
    "Figure Name": ["Overall"],
    "MAE": [overall_mae],
    "RMSE": [overall_rmse],
    "Inference Speed (samples/second)": [overall_avg_speed]
})
error_df = pd.concat([error_df, overall_row], ignore_index=True)

# Save the error metrics to an Excel file
error_save_path = os.path.join(SAVE_FIGURE_DIR, "Error_Saved.xlsx")
with pd.ExcelWriter(error_save_path, engine='xlsxwriter') as writer:
    error_df.to_excel(writer, index=False)
print("All figures and error metrics saved successfully.")

# Plot and save bar charts for the errors
num_plots = (len(error_df) - 1) // 10 + 1
for i in range(num_plots):
    start_idx = i * 10
    end_idx = min((i + 1) * 10, len(error_df) - 1)
    fig, ax = plt.subplots(figsize=(12, 8))
    error_df[start_idx:end_idx].plot(kind='bar', x='Figure Name', y=['MAE', 'RMSE'], ax=ax)
    plt.xticks(rotation=90)
    plt.title(f'MAE and RMSE for each interval (Part {i + 1})')
    plt.xlabel('Figure Name')
    plt.ylabel('Error')
    plt.tight_layout()
    error_bar_plot_name = f"Error_Bar_Plot_Part_{i + 1}.png"
    plt.savefig(os.path.join(FIGURE_ERROR_DIR, error_bar_plot_name))
    plt.close(fig)
    print(f"Saved error bar plot {error_bar_plot_name} in {FIGURE_ERROR_DIR}")

print(f"Completed processing {num_intervals} intervals.")

print("\n" + "="*50)
print("PROCESSING COMPLETED!")
print("="*50)
print(f"✅ Processed {num_intervals} intervals")
print(f"✅ Generated action figures: {num_intervals}")
print(f"✅ Generated state figures: {num_intervals}")
print(f"✅ Generated intermediate error plots: {num_intervals}")
print(f"✅ Generated final error plots: {(len(error_df) - 1) // 10 + 1}")
print(f"✅ Saved Excel error report: Error_Saved.xlsx")
print(f"\nAll figures saved in: {SAVE_FIGURE_DIR}")
print(f"- Action figures: {FIGURE_ACTION_DIR}")
print(f"- State figures: {FIGURE_state_DIR}")
print(f"- Error figures: {FIGURE_ERROR_DIR}")
print("="*50)
