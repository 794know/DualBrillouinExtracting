# G_TrainingProcessConfig.py
# Author: QYH
# Version: 1.1
# Date: 2025/05/29
# This code is used for training the dual-channel CNN model:
# curve dataset with dual channels and labels
# All the index can be modified in 'A_fiber_index.py'

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from E_DatasetConfig import PumpPowerDataset
from F_DualChannelCNN import DualChannelCNN
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端，不显示图像
import matplotlib.pyplot as plt
from tqdm import tqdm
import time  # 导入时间模块


if __name__ == "__main__":
    
    print("This script is intended to be run directly.")
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir='runs/dual_channel_experiment')  # 新增

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
    print(f"Dataset size: {len(dataset)}")  # 输出数据集大小

    # 划分训练集、验证集和测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型实例并移至设备
    model = DualChannelCNN().to(device)
    print("Model created and moved to device.")
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()  # 均方误差损失，适用于回归任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 用于存储损失值以便可视化
    train_losses = []
    val_losses = []
    train_loss1_values = []
    train_loss2_values = []
    val_loss1_values = []
    val_loss2_values = []

    # 等待用户确认
    input("Press Enter to start training...")

    # 训练模型
    num_epochs = 100  # 设置训练轮数

    target_loss1 = 1  # 设置目标损失值1
    target_loss2 = 1  # 设置目标损失值2
    target_loss_total = 3  # 设置目标总损失值

    total_start_time = time.time()  # 记录总训练时间的开始
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_loss1 = 0.0
        running_train_loss2 = 0.0
        epoch_start_time = time.time()  # 记录当前 epoch 的开始时间
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for clean, distorted, label in progress_bar:
            # 将数据移至设备
            clean, distorted, label = clean.to(device), distorted.to(device), label.to(device)
            print(f"Batch size: {clean.size(0)}")
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = model(clean, distorted)
            
            # 分别计算两个标签的损失
            loss1 = criterion(output[:, 0], label[:, 0])
            loss2 = criterion(output[:, 1], label[:, 1])
            loss1 = torch.sqrt(loss1)  # 对损失1取平方根
            loss2 = torch.sqrt(loss2) * 0.05  # 对损失2取平方根

            # 计算总损失
            loss_total = loss1 + loss2
            
            # 反向传播和优化
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            running_train_loss += loss_total.item()
            running_train_loss1 += loss1.item()
            running_train_loss2 += loss2.item()

            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss_total.item():.4f}"})
        
        # 记录每个epoch的平均训练损失
        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_loss1 = running_train_loss1 / len(train_loader)
        avg_train_loss2 = running_train_loss2 / len(train_loader)
        # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_loss1', avg_train_loss1, epoch)
        writer.add_scalar('Loss/train_loss2', avg_train_loss2, epoch)
        train_losses.append(avg_train_loss)
        train_loss1_values.append(avg_train_loss1)
        train_loss2_values.append(avg_train_loss2)

        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        running_val_loss1 = 0.0
        running_val_loss2 = 0.0
        with torch.no_grad():
            for clean, distorted, label in val_loader:
                clean, distorted, label = clean.to(device), distorted.to(device), label.to(device)
                output = model(clean, distorted)
                loss1 = criterion(output[:, 0], label[:, 0])
                loss2 = criterion(output[:, 1], label[:, 1])
                loss1 = torch.sqrt(loss1)  # 对损失1取平方根
                loss2 = torch.sqrt(loss2) * 0.05  # 对损失2取平方根
                loss_total = loss1 + loss2
                running_val_loss += loss_total.item()
                running_val_loss1 += loss1.item()
                running_val_loss2 += loss2.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_loss1 = running_val_loss1 / len(val_loader)
        avg_val_loss2 = running_val_loss2 / len(val_loader)
        # 记录验证损失到TensorBoard
        writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
        writer.add_scalar('Loss/val_loss1', avg_val_loss1, epoch)
        writer.add_scalar('Loss/val_loss2', avg_val_loss2, epoch)
        val_losses.append(avg_val_loss)
        val_loss1_values.append(avg_val_loss1)
        val_loss2_values.append(avg_val_loss2)
        
        epoch_end_time = time.time()  # 记录当前 epoch 的结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算当前 epoch 的持续时间
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_duration:.2f}s")

        if avg_val_loss <= target_loss_total:
            print(f'Reached target loss of {target_loss_total} at epoch {epoch+1}')
            break

    # tensorboard writer 关闭
    writer.close()

    total_end_time = time.time()  # 记录总训练时间的结束
    total_duration = total_end_time - total_start_time  # 计算总训练时间
    print(f"Total training time: {total_duration:.2f}s")

    # 保存模型
    torch.save(model.state_dict(), "model.pth")

    # 测试阶段
    model.eval()
    running_test_loss = 0.0
    running_test_loss1 = 0.0
    running_test_loss2 = 0.0
    with torch.no_grad():
        for clean, distorted, label in test_loader:
            clean, distorted, label = clean.to(device), distorted.to(device), label.to(device)
            output = model(clean, distorted)
            loss1 = criterion(output[:, 0], label[:, 0])
            loss2 = criterion(output[:, 1], label[:, 1])
            loss1 = torch.sqrt(loss1)  # 对损失1取平方根
            loss2 = torch.sqrt(loss2) * 0.05  # 对损失2取平方根
            loss_total = loss1 + loss2
            running_test_loss += loss_total.item()
            running_test_loss1 += loss1.item()
            running_test_loss2 += loss2.item()
            
    avg_test_loss = running_test_loss / len(test_loader)
    avg_test_loss1 = running_test_loss1 / len(test_loader)
    avg_test_loss2 = running_test_loss2 / len(test_loader)
    
    print(f"Test Loss: {avg_test_loss:.4f}")

    # 可视化损失曲线并保存为图像文件
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_loss1_values, label='Train Loss1')
    plt.plot(val_loss1_values, label='Val Loss1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss1 Over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_loss2_values, label='Train Loss2')
    plt.plot(val_loss2_values, label='Val Loss2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss2 Over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='Train Loss Total')
    plt.plot(val_losses, label='Val Loss Total')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Total Over Epochs')
    plt.legend()


    plt.tight_layout()
    plt.savefig("losses_over_epochs.png")  # 保存为图像文件
    print("Losses plot saved as 'losses_over_epochs.png'")