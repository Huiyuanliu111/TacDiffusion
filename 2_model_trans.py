import torch
from act.policy import ACTPolicy
import sys
import multiprocessing

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args_override():
    return {
        'ckpt_dir': 'checkpoints',
        'seed': 42,
        'num_epochs': 300,
        'lr': 5e-5,
        'hidden_dim': 512,
        'kl_weight': 30.0,
        'num_queries': 200,
        'dropout': 0.1,
    }

def main():
    args_override = get_args_override()

    # 初始化模型
    model = ACTPolicy(args_override).to(device)

    # 加载权重
    model.load_state_dict(torch.load("output/best_ACT.pth", map_location=device))
    model.eval()

    # 定义 batch size
    batch_size = 32

    # 构造 dummy 输入
    dummy_qpos = torch.randn(batch_size, 18).to(device)  # [batch_size, 18]
    dummy_image = torch.zeros((batch_size, 3, 224, 224)).to(device)  # [batch_size, 3, 224, 224]

    # 导出ONNX
    input_names = ["qpos", "image"]
    output_names = ["output"]
    torch.onnx.export(
        model,
        (dummy_qpos,dummy_image),
        "output/ACT.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "qpos": {0: "batch_size"},
            "image": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
