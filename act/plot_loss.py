import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def read_tensorboard_logs(log_dir):
    """读取TensorBoard日志文件并提取训练和验证损失"""
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # 获取可用的标量标签
        tags = event_acc.Tags()['scalars']
        print(f"可用的标量标签: {tags}")
        
        train_loss_data = []
        val_loss_data = []
        
        # 提取训练损失
        if 'training_loss' in tags:
            train_scalars = event_acc.Scalars('training_loss')
            train_loss_data = [(s.step, s.value) for s in train_scalars]
        
        # 提取验证损失
        if 'validation_loss' in tags:
            val_scalars = event_acc.Scalars('validation_loss')
            val_loss_data = [(s.step, s.value) for s in val_scalars]
        
        return train_loss_data, val_loss_data
    except Exception as e:
        print(f"读取日志 {log_dir} 时出错: {e}")
        return [], []

def plot_loss_curves(train_data, val_data, save_path=None, title="训练损失曲线"):
    """绘制损失曲线"""
    plt.figure(figsize=(14, 8))
    
    # 绘制训练损失
    if train_data:
        train_steps, train_losses = zip(*train_data)
        plt.plot(train_steps, train_losses, label='训练损失', alpha=0.6, linewidth=1, color='blue')
        
        # 计算移动平均以平滑曲线
        if len(train_losses) > 50:
            window_size = max(10, len(train_losses) // 100)  # 动态窗口大小
            train_losses_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            train_steps_smooth = train_steps[window_size-1:]
            plt.plot(train_steps_smooth, train_losses_smooth, label='训练损失(平滑)', linewidth=2, color='darkblue')
    
    # 绘制验证损失
    if val_data:
        val_steps, val_losses = zip(*val_data)
        plt.plot(val_steps, val_losses, label='验证损失', marker='o', linewidth=2, markersize=3, color='red')
    
    plt.xlabel('训练步数 (Steps)', fontsize=12)
    plt.ylabel('损失值 (Loss)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴为对数刻度以便更好地观察损失变化
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线已保存到: {save_path}")
    
    plt.show()

def get_latest_log_dir(base_log_dir="logs/fit"):
    """获取最新的日志目录"""
    log_dirs = glob.glob(os.path.join(base_log_dir, "*"))
    if not log_dirs:
        raise ValueError(f"在 {base_log_dir} 中未找到日志目录")
    
    # 按修改时间排序，获取最新的
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_dirs[0]

def list_available_logs(base_log_dir="logs/fit"):
    """列出所有可用的日志目录"""
    log_dirs = glob.glob(os.path.join(base_log_dir, "*"))
    if not log_dirs:
        print(f"在 {base_log_dir} 中未找到日志目录")
        return []
        
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print("可用的训练日志:")
    for i, log_dir in enumerate(log_dirs[:10]):  # 只显示最新的10个
        dir_name = os.path.basename(log_dir)
        mtime = datetime.fromtimestamp(os.path.getmtime(log_dir))
        print(f"{i+1:2d}. {dir_name} (修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return log_dirs

def main():
    """主函数"""
    try:
        # 列出可用的日志
        available_logs = list_available_logs()
        
        if not available_logs:
            print("未找到任何训练日志。请先运行训练脚本。")
            return
        
        # 使用最新的日志目录
        log_dir = get_latest_log_dir()
        print(f"\n使用最新的日志目录: {log_dir}")
        
        # 读取TensorBoard日志
        train_data, val_data = read_tensorboard_logs(log_dir)
        
        if not train_data and not val_data:
            print("未找到损失数据。请检查日志目录是否包含有效的TensorBoard数据。")
            return
        
        print(f"\n数据统计:")
        print(f"训练损失数据点: {len(train_data)}")
        print(f"验证损失数据点: {len(val_data)}")
        
        # 绘制损失曲线
        dir_name = os.path.basename(log_dir)
        title = f"ACT模型训练损失曲线\n{dir_name}"
        save_path = "figures/loss_curves.png"
        plot_loss_curves(train_data, val_data, save_path=save_path, title=title)
        
        # 打印一些统计信息
        print(f"\n损失统计:")
        if train_data:
            final_train_loss = train_data[-1][1]
            min_train_loss = min(loss for _, loss in train_data)
            print(f"最终训练损失: {final_train_loss:.6f}")
            print(f"最小训练损失: {min_train_loss:.6f}")
        
        if val_data:
            final_val_loss = val_data[-1][1]
            min_val_loss = min(loss for _, loss in val_data)
            print(f"最终验证损失: {final_val_loss:.6f}")
            print(f"最小验证损失: {min_val_loss:.6f}")
    
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 