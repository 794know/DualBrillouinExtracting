# F_DualChannelCNN.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/26
# This code is used for defining a dual-channel CNN model for training:
# Model for processing dual-channel data (e.g., temperature and strain curves)
# All the index can be modified in 'A_fiber_index.py'

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualChannelCNN(nn.Module):
    def __init__(self):
        super(DualChannelCNN, self).__init__()
        # 第一个通道的卷积层
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第二个通道的卷积层
        self.conv2_1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 合并后的全连接层
        self.fc1 = nn.Linear(32 * 150 * 2, 128)  # 假设经过池化后长度变为150
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 输出温度和应力值

    def forward(self, x1, x2):
        # 第一个通道的前向传播
        x1 = F.relu(self.conv1_1(x1))
        x1 = self.pool1(F.relu(self.conv1_2(x1)))
        
        # 第二个通道的前向传播
        x2 = F.relu(self.conv2_1(x2))
        x2 = self.pool2(F.relu(self.conv2_2(x2)))
        
        # 将两个通道的特征合并
        x1 = x1.view(x1.size(0), -1)  # 展平
        x2 = x2.view(x2.size(0), -1)  # 展平
        x = torch.cat((x1, x2), dim=1)  # 按维度1拼接
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = DualChannelCNN()
    
    # 创建两个输入张量，假设输入曲线长度为600
    input1 = torch.randn(1, 1, 600)  # 第一个通道的输入
    input2 = torch.randn(1, 1, 600)  # 第二个通道的输入
    
    # 前向传播
    output = model(input1, input2)
    print(output.shape)  # 输出形状，例如：torch.Size([1, 2])