# 将ACT算法融入到本项目内

## 与原ACT项目的区别：

删除视觉部分（image, camera, backbone）
  删除CNNMLP的选项
  将backbone和camera_names默认值设为None和空列表，这样可以最小限度地改变原有模型

修改输入输出维度
  新输入维度：
    observation: bs * 36
    F: bs * 6
  原ACT输入维度：
    joint position: bs * 14
    action prediction: bs * num_queries * 14
    env_state: bs 

## 项目结构

`transformer.py` - 实现transformer的编码器和解码器
`detr_vae.py` - 实现基于DETR架构的VAE模型，包含编码器和解码器的核心实现
`imitate_episodes.py` - 主要训练和评估脚本，包含数据加载、模型训练、验证和评估的完整流程
`__init__.py` - 模块初始化文件，提供模型构建接口



