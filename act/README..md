# 将ACT算法融入到本项目内

## 与原ACT项目的区别：

删除视觉部分（image, camera, backbone）
  删除CNNMLP的选项
  将backbone和camera_names默认值设为None和空列表，这样可以最小限度地改变原有模型

修改输入输出维度
  新输入维度：
    observation: bs * 18 也就是只有 opresent  当然之后可以尝试加oprev
    F: bs * 6
  原ACT输入维度：
    joint position: bs * 14
    action prediction: bs * num_queries * 14
    env_state: bs 

修改data_split.py
  用于action chunking提取多个时间步的数据
  将RobotCustomDataset模仿ACT的episodic dataset，包含多个episode。每次返回的是取样的episode内的随机时间点。
修改1_model_train.py
  action chunking

## 项目结构

`transformer.py` - 实现transformer的编码器和解码器
`detr_vae.py` - 实现基于DETR架构的VAE模型，包含编码器和解码器的核心实现
`imitate_episodes.py` - 原ACT项目中用以主要训练和评估脚本，此处不使用
`policy.py ` - 实现ACT Policy

## 使用方法
运行`python 1_model_train.py`




