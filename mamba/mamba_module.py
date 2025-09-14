import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .mamba2 import Mamba2

class RoboM2TEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 6, d_state: int = 16, dim_feedforward: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state)
            for _ in range(num_layers)
        ])
        
        # Feed Forward层
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            )
            for _ in range(num_layers)
        ])
        
        # 归一化层
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers * 2)  # 每层两个归一化操作
        ])
        
    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        for i, (ff, mamba) in enumerate(zip(self.feed_forward, self.layers)):
            # Mamba层
            mamba_out = mamba(x)
            x = x + mamba_out  # 第一个残差连接
            x = self.norms[i*2](x)  # 第一个归一化
            
            # Feed Forward层
            ff_out = ff(x)
            x = x + ff_out  # 第二个残差连接
            x = self.norms[i*2+1](x)  # 第二个归一化
        return x

class RoboM2TDecoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 6, d_state: int = 16, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state)
            for _ in range(num_layers)
        ])
        
        # Cross Attention层
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead)
            for _ in range(num_layers)
        ])
        
        # Feed Forward层
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            )
            for _ in range(num_layers)
        ])
        
        # 归一化层
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers * 3)  # 每层三个归一化操作
        ])

        
    def forward(self, x: torch.Tensor, memory: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        # memory: [batch_size, seq_len, d_model]
        for i, (mamba, cross_attn, ff) in enumerate(zip(self.layers, self.cross_attention, self.feed_forward)):
            # Mamba层
            mamba_out = mamba(x)
            x = x + mamba_out  # 第一个残差连接
            x = self.norms[i*3](x)  # 第一个归一化
            
            # Cross Attention (需要转换维度顺序)
            x_transposed = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
            memory_transposed = memory.transpose(0, 1)  # [seq_len, batch_size, d_model]
            cross_attn_out = cross_attn(x_transposed, memory_transposed, memory_transposed)[0]
            cross_attn_out = cross_attn_out.transpose(0, 1)  # [batch_size, seq_len, d_model]
            x = x + cross_attn_out  # 第二个残差连接
            x = self.norms[i*3+1](x)  # 第二个归一化
            
            # Feed Forward
            ff_out = ff(x)
            x = x + ff_out  # 第三个残差连接
            x = self.norms[i*3+2](x)  # 第三个归一化
        return x

class RoboM2T(nn.Module):
    def __init__(self, d_model: int = 512, num_layers: int = 6, d_state: int = 16, nhead: int = 8):
        super().__init__()
        self.encoder = RoboM2TEncoder(d_model, num_layers, d_state)
        self.decoder = RoboM2TDecoder(d_model, num_layers, d_state, nhead)
        
        # 输入和输出投影层
        self.state_proj = nn.Linear(18, d_model)  # 状态投影层
        self.action_proj = nn.Linear(6, d_model)  # 动作投影层
        self.output_proj = nn.Linear(d_model, 6)  # 输出投影层
        
        # Add & Norm层
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 输入投影
        src = self.state_proj(src)  # [batch_size, seq_len, d_model]
        src = self.norm(src)
        
        # 编码器
        memory = self.encoder(src, pos)  # [batch_size, seq_len, d_model]
        
        # 目标序列投影
        tgt = self.action_proj(tgt)  # [batch_size, seq_len, d_model]
        tgt = self.norm(tgt)
        
        # 解码器
        decoder_output = self.decoder(tgt, memory, pos)  # [batch_size, seq_len, d_model]
        
        # 输出投影
        action = self.output_proj(decoder_output)  # [batch_size, seq_len, 6]
        
        return action