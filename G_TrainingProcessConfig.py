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
import matplotlib.pyplot as plt
from F_DualChannelCNN import DualChannelCNN
from E_DatasetConfig import PumpPowerDataset

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# 创建模型实例并移至设备
model = DualChannelCNN().to(device)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()  # 均方误差损失，适用于回归任务
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 用于存储损失值以便可视化
losses = []

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (clean, distorted, label) in enumerate(train_loader):
        # 将数据移至设备
        clean, distorted, label = clean.to(device), distorted.to(device), label.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(clean, distorted)
        
        # 分别计算两个标签的损失
        loss1 = criterion(output[:, 0], label[:, 0])
        loss2 = criterion(output[:, 1], label[:, 1])
        
        # 计算总损失
        loss_total = 0.5 * loss1 + 0.5 * 0.05 * loss2
        
        # 反向传播和优化
        loss_total.backward()
        optimizer.step()
        
        running_loss += loss_total.item()
    
    # 记录每个epoch的平均损失
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "model.pth")

# 可视化损失曲线
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()