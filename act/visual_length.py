import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATASET_PATH = "dataset"
state_dataset = 'sensor_all.pkl'

# 创建figures目录（如果不存在）
FIGURES_PATH = "figures"
os.makedirs(FIGURES_PATH, exist_ok=True)

# 加载状态数据
pkl_file_path_state = os.path.join(DATASET_PATH, state_dataset)

try:
    with open(pkl_file_path_state, 'rb') as f:
        state_all = pickle.load(f)
    
    print(f"总episode数量: {len(state_all)}")
    
    # 获取每个episode的长度
    episode_lengths = [episode.shape[0] for episode in state_all]
    
    # 基本统计信息
    print(f"Episode长度统计:")
    print(f"  最小长度: {min(episode_lengths)}")
    print(f"  最大长度: {max(episode_lengths)}")
    print(f"  平均长度: {np.mean(episode_lengths):.2f}")
    print(f"  中位数长度: {np.median(episode_lengths):.2f}")
    print(f"  标准差: {np.std(episode_lengths):.2f}")
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Episode长度分布可视化', fontsize=16)
    
    # 1. 直方图
    axes[0, 0].hist(episode_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Episode长度直方图')
    axes[0, 0].set_xlabel('Episode长度')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 箱线图
    axes[0, 1].boxplot(episode_lengths, patch_artist=True, 
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[0, 1].set_title('Episode长度箱线图')
    axes[0, 1].set_ylabel('Episode长度')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Episode索引 vs 长度散点图
    axes[1, 0].scatter(range(len(episode_lengths)), episode_lengths, 
                       alpha=0.6, s=20, color='red')
    axes[1, 0].set_title('Episode索引 vs 长度')
    axes[1, 0].set_xlabel('Episode索引')
    axes[1, 0].set_ylabel('Episode长度')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 累积分布函数
    sorted_lengths = np.sort(episode_lengths)
    cumulative_prob = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    axes[1, 1].plot(sorted_lengths, cumulative_prob, linewidth=2, color='purple')
    axes[1, 1].set_title('累积分布函数')
    axes[1, 1].set_xlabel('Episode长度')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 保存图片到figures目录
    plt.savefig(os.path.join(FIGURES_PATH, 'episode_lengths_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细的长度分布
    print(f"\n前10个episode的长度:")
    for i in range(min(10, len(episode_lengths))):
        print(f"  Episode {i}: {episode_lengths[i]}")
    
    # 找出异常长或短的episode
    print(f"\n最短的5个episode:")
    shortest_indices = np.argsort(episode_lengths)[:5]
    for idx in shortest_indices:
        print(f"  Episode {idx}: {episode_lengths[idx]} steps")
    
    print(f"\n最长的5个episode:")
    longest_indices = np.argsort(episode_lengths)[-5:]
    for idx in longest_indices:
        print(f"  Episode {idx}: {episode_lengths[idx]} steps")
    
    # 计算不同长度范围的episode数量
    length_ranges = [
        (0, 1000),
        (1000, 5000),
        (5000, 10000),
        (10000, 15000),
        (15000, float('inf'))
    ]
    
    print(f"\nEpisode长度分布:")
    for min_len, max_len in length_ranges:
        if max_len == float('inf'):
            count = sum(1 for length in episode_lengths if length >= min_len)
            print(f"  {min_len}+ steps: {count} episodes")
        else:
            count = sum(1 for length in episode_lengths if min_len <= length < max_len)
            print(f"  {min_len}-{max_len-1} steps: {count} episodes")

except FileNotFoundError:
    print(f"错误: 找不到文件 '{pkl_file_path_state}'")
except Exception as e:
    print(f"错误: {e}") 