# G_TrainingProcessConfig.py
# Author: QYH
# Version: 1.0
# Date: 2025/05/26
# This code is used for training the dual-channel CNN model:
# curve dataset with dual channels and labels
# All the index can be modified in 'A_fiber_index.py'

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from F_DualChannelCNN import DualChannelCNN
from E_DatasetConfig import PumpPowerDataset

# 数据集路径列表
data_dirs = [
    "dataset_clean",
    "dataset_SNR_6.0dB",
    "dataset_SNR_9.0dB",
    "dataset_SNR_12.0dB",
    "dataset_SNR_15.0dB"
]

# 创建数据集实例
dataset = PumpPowerDataset(data_dirs)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型实例
model = DualChannelCNN()

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()  # 均方误差损失，适用于回归任务
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (clean, distorted, label) in enumerate(train_loader):
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(clean, distorted)
        
        # 计算损失
        loss = criterion(output, label)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "model.pth")
# 输出训练完成信息
print("Training completed and model saved as 'model.pth'.")