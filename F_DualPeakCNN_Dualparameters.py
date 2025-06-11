# F_DualPeakCNN_Dualparameters.py
# Author: QYH
# Version: 2.0
# Date: 2025/06/11
# This code is used for defining a dual-channel CNN model for training:
# Model for processing dual-channel data (e.g., temperature and strain curves)
# All the index can be modified in 'A_fiber_index.py'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DualPeakCNN(nn.Module):
    def __init__(self, input_dim=600):
        super().__init__()
        
        # 1. 浅层共享特征
        self.shallow = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        
        # 2. 动态双峰分离器
        self.peak_splitter = nn.Sequential(
            nn.Conv1d(32, 16, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 2, 5, padding=2),
            nn.Softmax(dim=1)
        )
        
        # 3. 双分支特征精炼
        self.peak1_refine = nn.Sequential(
            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        self.peak2_refine = nn.Sequential(
            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        # 4. 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 5. 回归头（修改为延迟初始化）
        self.reg_head = None  # 先占位
        self._init_reg_head()  # 单独初始化方法

    def _init_reg_head(self):
        self.reg_head = nn.Linear(128, 2)
        nn.init.xavier_normal_(self.reg_head.weight, gain=1.0)
        # 手动缩放每一列
        with torch.no_grad():
            self.reg_head.weight[:, 0] *= 0.08
            self.reg_head.weight[:, 1] *= 0.016
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, x):
        # x = x.unsqueeze(1)
        shared = self.shallow(x)
        attn = self.peak_splitter(shared)
        
        peak1 = self.peak1_refine(shared * attn[:, 0:1])
        peak2 = self.peak2_refine(shared * attn[:, 1:2])
        
        fused = self.fusion(torch.cat([peak1, peak2], dim=1))
        return self.reg_head(fused)

if __name__ == '__main__':
    model = DualPeakCNN()

    # 打印模型结构
    print(model)
    
    # 使用 torchsummary 打印详细结构
    summary(model, (1, 600), device='cpu')