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
from act.temporal_agg import temporal_aggregation


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
error_df = pd.DataFrame(columns=["Figure Name", "MAE", "RMSE", "Inference Speed (samples/second)", "Method"])

# Lists to store all ground truth, predictions, and inference speeds for overall error calculation
all_y_true = []
all_y_pred_temporal = []  # 使用temporal aggregation的预测
all_y_pred_direct = []    # 不使用temporal aggregation的预测
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
    y_pred_no_temporal = np.zeros((interval_length, y_dim))  # 存储不使用temporal agg的预测结果
    y_pred_temporal = np.zeros((interval_length, y_dim)) # 存储使用temporal agg的预测结果

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
            
            # 存储不使用temporal agg的直接预测结果
            y_pred_no_temporal[i] = all_actions[0].numpy()
            
            if temporal_agg:
                raw_action = temporal_aggregation(i, all_actions, all_time_actions, num_queries, k)
            else:
                raw_action = all_actions[0:1]
            
            y_pred_temporal[i] = raw_action.cpu().numpy().squeeze(0)

            # 存储结果
            y_pred[i] = all_actions.numpy()  # 保存完整序列用于可视化

    end_time = time.time()
    inference_time = end_time - start_time
    inference_speeds = interval_length / inference_time
    print(f"Interval {interval_idx}: inference speeds: {inference_speeds} sample/second")

    # Calculate errors for the current interval
    y_true_interval = y_true_original[start_idx:end_idx]

    # 反标准化预测结果
    if hasattr(torch_data_test, 'denormalize_actions'):
        y_pred_temporal_original = torch_data_test.denormalize_actions(y_pred_temporal)
        y_pred_no_temporal_original = torch_data_test.denormalize_actions(y_pred_no_temporal)
    else:
        y_pred_temporal_original = y_pred_temporal
        y_pred_no_temporal_original = y_pred_no_temporal

    # 计算两种方法的误差
    mae_temporal = mean_absolute_error(y_true_interval, y_pred_temporal_original)
    rmse_temporal = np.sqrt(mean_squared_error(y_true_interval, y_pred_temporal_original))
    
    mae_direct = mean_absolute_error(y_true_interval, y_pred_no_temporal_original)
    rmse_direct = np.sqrt(mean_squared_error(y_true_interval, y_pred_no_temporal_original))

    # Add results to DataFrame
    figure_saved_name = f"action_data_{start_idx}_{end_idx}.png"
    new_rows = pd.DataFrame({
        "Figure Name": [figure_saved_name, figure_saved_name],
        "MAE": [mae_temporal, mae_direct],
        "RMSE": [rmse_temporal, rmse_direct],
        "Inference Speed (samples/second)": [inference_speeds, inference_speeds],
        "Method": ["Temporal Agg", "Direct Pred"]
    })
    error_df = pd.concat([error_df, new_rows], ignore_index=True)

    # Add to the lists for overall error calculation
    all_y_true.append(y_true_interval)
    all_y_pred_temporal.append(y_pred_temporal_original)
    all_y_pred_direct.append(y_pred_no_temporal_original)
    all_inference_speeds.append(inference_speeds)

    # Plot and save the action data figures with comparison
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    
    for i in range(6):
        row, col = divmod(i, 2)
        # Plot with temporal aggregation
        axs[row, col].plot(y_pred_temporal_original[:, i], 
                          label=f'{label_pred[i]} (Temporal)', 
                          linestyle='-', 
                          color='blue')
        # Plot without temporal aggregation
        axs[row, col].plot(y_pred_no_temporal_original[:, i], 
                          label=f'{label_pred[i]} (Direct)', 
                          linestyle='--', 
                          color='green')
        # Plot ground truth
        axs[row, col].plot(y_true_interval[:, i], 
                          label=label_true[i], 
                          linestyle=':', 
                          color='red')

        axs[row, col].set_xlabel('Index')
        axs[row, col].set_ylabel('Value')
        axs[row, col].set_title(f'Comparison of Predictions - {label_pred[i]}')
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.suptitle(f'Temporal Aggregation vs Direct Prediction\nMAE: {mae_temporal:.4f} vs {mae_direct:.4f}, RMSE: {rmse_temporal:.4f} vs {rmse_direct:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_ACTION_DIR, figure_saved_name))
    plt.close(fig)

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

# Calculate overall error metrics for both methods
all_y_true = np.vstack(all_y_true)
all_y_pred_temporal = np.vstack(all_y_pred_temporal)
all_y_pred_direct = np.vstack(all_y_pred_direct)

overall_mae_temporal = mean_absolute_error(all_y_true, all_y_pred_temporal)
overall_rmse_temporal = np.sqrt(mean_squared_error(all_y_true, all_y_pred_temporal))
overall_mae_direct = mean_absolute_error(all_y_true, all_y_pred_direct)
overall_rmse_direct = np.sqrt(mean_squared_error(all_y_true, all_y_pred_direct))
overall_avg_speed = np.mean(all_inference_speeds)

# Add overall metrics to DataFrame
overall_rows = pd.DataFrame({
    "Figure Name": ["Overall", "Overall"],
    "MAE": [overall_mae_temporal, overall_mae_direct],
    "RMSE": [overall_rmse_temporal, overall_rmse_direct],
    "Inference Speed (samples/second)": [overall_avg_speed, overall_avg_speed],
    "Method": ["Temporal Agg", "Direct Pred"]
})
error_df = pd.concat([error_df, overall_rows], ignore_index=True)

# Save the error metrics to an Excel file with separate sheets for each method
error_save_path = os.path.join(SAVE_FIGURE_DIR, "Error_Comparison.xlsx")
with pd.ExcelWriter(error_save_path, engine='xlsxwriter') as writer:
    # Save all results
    error_df.to_excel(writer, sheet_name='All Results', index=False)
    
    # Save temporal aggregation results
    temporal_results = error_df[error_df['Method'] == 'Temporal Agg']
    temporal_results.to_excel(writer, sheet_name='Temporal Agg', index=False)
    
    # Save direct prediction results
    direct_results = error_df[error_df['Method'] == 'Direct Pred']
    direct_results.to_excel(writer, sheet_name='Direct Pred', index=False)

# Plot and save comparison bar charts
num_plots = (len(error_df) - 2) // 20 + 1  # -2 for overall results, 20 samples per plot
for i in range(num_plots):
    start_idx = i * 20
    end_idx = min((i + 1) * 20, len(error_df) - 2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot MAE comparison
    data_slice = error_df[start_idx:end_idx]
    temporal_data = data_slice[data_slice['Method'] == 'Temporal Agg']
    direct_data = data_slice[data_slice['Method'] == 'Direct Pred']
    
    x = np.arange(len(temporal_data))
    width = 0.35
    
    ax1.bar(x - width/2, temporal_data['MAE'], width, label='Temporal Agg')
    ax1.bar(x + width/2, direct_data['MAE'], width, label='Direct Pred')
    ax1.set_ylabel('MAE')
    ax1.set_title('MAE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(temporal_data['Figure Name'], rotation=45)
    ax1.legend()
    
    # Plot RMSE comparison
    ax2.bar(x - width/2, temporal_data['RMSE'], width, label='Temporal Agg')
    ax2.bar(x + width/2, direct_data['RMSE'], width, label='Direct Pred')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(temporal_data['Figure Name'], rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_ERROR_DIR, f'Error_Comparison_Part_{i+1}.png'))
    plt.close(fig)

print("\n" + "="*50)
print("PROCESSING COMPLETED!")
print("="*50)
print(f"✅ Processed {num_intervals} intervals")
print(f"✅ Generated comparison figures: {num_intervals}")
print(f"✅ Generated state figures: {num_intervals}")
print(f"✅ Generated error comparison plots: {num_plots}")
print(f"✅ Saved Excel comparison report: Error_Comparison.xlsx")
print("\nOverall Results:")
print(f"Temporal Aggregation: MAE={overall_mae_temporal:.4f}, RMSE={overall_rmse_temporal:.4f}")
print(f"Direct Prediction: MAE={overall_mae_direct:.4f}, RMSE={overall_rmse_direct:.4f}")
print(f"\nAll figures saved in: {SAVE_FIGURE_DIR}")
print(f"- Action comparison figures: {FIGURE_ACTION_DIR}")
print(f"- State figures: {FIGURE_state_DIR}")
print(f"- Error comparison figures: {FIGURE_ERROR_DIR}")
print("="*50)
