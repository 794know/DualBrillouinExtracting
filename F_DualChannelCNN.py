# F_DualChannelCNN.py
# Author: QYH
# Version: 1.1
# Date: 2025/05/27
# This code is used for defining a dual-channel CNN model for training:
# Model for processing dual-channel data (e.g., temperature and strain curves)
# All the index can be modified in 'A_fiber_index.py'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DualChannelCNN(nn.Module):
    def __init__(self):
        super(DualChannelCNN, self).__init__()
        
        # 通道1的卷积层
        self.conv1_channel1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.bn1_channel1 = nn.BatchNorm1d(8)
        self.pool1_channel1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2_channel1 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.bn2_channel1 = nn.BatchNorm1d(16)
        self.pool2_channel1 = nn.MaxPool1d(kernel_size=2)
        
        # 通道1的注意力机制
        self.attention_channel1 = ChannelAttention(num_channels=16)
        
        # 通道2的卷积层
        self.conv1_channel2 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.bn1_channel2 = nn.BatchNorm1d(8)
        self.pool1_channel2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2_channel2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.bn2_channel2 = nn.BatchNorm1d(16)
        self.pool2_channel2 = nn.MaxPool1d(kernel_size=2)
        
        # 计算全连接层输入维度
        flattened_size = 16 * 150  # 输入长度为600，经过两次池化后变为150，再乘以通道数16
        
        # 共享全连接层
        self.fc1 = nn.Linear(flattened_size * 2, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)  # 输出2个标量

    def forward(self, x1, x2):
        # 通道1的特征提取
        x1 = self.pool1_channel1(F.relu(self.bn1_channel1(self.conv1_channel1(x1))))
        x1 = self.pool2_channel1(F.relu(self.bn2_channel1(self.conv2_channel1(x1))))
        
        # 应用注意力机制
        x1 = self.attention_channel1(x1)
        
        # 通道2的特征提取
        x2 = self.pool1_channel2(F.relu(self.bn1_channel2(self.conv1_channel2(x2))))
        x2 = self.pool2_channel2(F.relu(self.bn2_channel2(self.conv2_channel2(x2))))
        
        # 展平特征
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        # 拼接双通道特征
        x = torch.cat([x1, x2], dim=1)
        
        # 共享全连接层
        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        x = self.fc2(x)
        
        return x

# This code is used for defining a dual-channel CNN model for training:
if __name__ == "__main__":
    # 创建模型实例
    model = DualChannelCNN()
    
    # 打印模型结构
    print(model)
    
    # 使用 torchsummary 打印详细结构
    # 假设输入数据的形状为 [batch_size, 1, 600]
    summary(model, [(1, 600), (1, 600)])