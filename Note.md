# Note

## 20250524

### 贵江Loss函数思路  

BGS全部归一化后添加噪声，噪声值用1除  

### 训练思路  

数据以及label全部放入压缩文件，网络1对应某训练集即可读取对应文件夹进行训练。  

## 20250526

### model试训1  

对应commit：20250526_2056_commit

#### 项目结构 20250526

project/  
├── dataset_clean/  
│   ├── Dataset_Pumppower_index_0_SNR_Clean.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_6.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_6.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_9.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_9.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_12.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_12.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_15.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_15.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── E_DatasetConfig.py              # 数据集定义  
├── F_DualChannelCNN.py             # 模型定义  
└── G_TrainingProcessConfig.py      # 训练脚本  

channel1输入：  
Dataset_Pumppower_index_0_SNR_Clean.npy, 大小[600*132300]

channel2输入：  
Pump_Power_index_0.npy, 大小[600*132300]

训练数据label：  
Label_Pumppower_index_0_SNR_Clean.npy, 大小[2*132300]

训练过程中发现，channel2需要与channel1数据集一一对应，这个功能需要在另外的网络中给训练集数据打标，该功能需要重新写一个.py文件进行，文件名：  
<C_2_pump_adding_to_dataset.py>  Testing single one

#### Loss函数设计

loss1，loss2，total_loss 均需要可视化。同时，可能需要根据两个label的scale进行label的归一化。  
学习目标，测试集以及验证集的划分需要进一步设置

发现错误：  
label的归一化应该按照最大值归一化，而不是按照每行归一化！  
明日任务：  
写好labels归一化的函数  

## 20250527

### model试训2

对应commit：20250527_1012_commit

#### 项目结构 20250527

project/  
├── dataset_clean/  
│   ├── Dataset_Pumppower_index_0_SNR_Clean.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_6.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_6.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_9.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_9.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_12.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_12.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── dataset_SNR_15.0dB/  
│   ├── Dataset_Pumppower_index_0_SNR_15.0dB.npy  
│   ├── Pump_Power_index_0.npy  
│   ├── Label_Pumppower_index_0_SNR_Clean.npy  
├── E_DatasetConfig.py              # 数据集定义  
├── F_DualChannelCNN.py             # 模型定义  
└── G_TrainingProcessConfig.py      # 训练脚本  

channel1输入：  
Dataset_Pumppower_index_0_SNR_Clean.npy, 大小[600*132300]

channel2输入：  
Pump_Power_index_0.npy, 大小[600*132300]

训练数据label：  
Label_Pumppower_index_0_SNR_Clean.npy, 大小[2*132300]

#### Label归一化函数的构建

label的原始结构为[2*132300]，进入dataloader后会被转置为[132300*2]，这样需要我对两列的最大值归一化  

#### 解释器环境配置

更新至python 3.9.21环境，原环境为pytorch。训练时仍然使用实验室电脑pytorch环境。训练epoch数量：30  
训练结果中温度应力loss下降均有一定趋势，但是validation loss比较大
在原本的训练文件中需要添加以下功能：

1. loss下降至所需目标
2. 网络drop训练并且冻结网络参数
3. 确定validation loss的下降方法，可能训练数据信噪比6 dB较低了感觉

### model试训3

对应commit：20250527_1612_commit

#### 策略

>多试试at vat的正则化。尽量把模型从监督学习转化为半监督模型，这样子模型会收敛到一个让你意想不到的效果，那就是模型拥有了对抗的能力，不会因为数据的微小变化，导致模型的效果跟着有很大变化。

训练后准备使用tensorboard可视化监控手段实时调控训练进度，避免网络训练导致的试错成本升高。

#### 网络结构更新

```text
DualChannelCNN(
  (conv1_channel1): Conv1d(1, 16, kernel_size=(3,), stride=(1,), padding=(1,))  
  (bn1_channel1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (pool1_channel1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  (conv2_channel1): Conv1d(16, 32, kernel_size=(3,), stride=(1,), padding=(1,))  
  (bn2_channel1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (pool2_channel1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  (conv1_channel2): Conv1d(1, 16, kernel_size=(3,), stride=(1,), padding=(1,))  
  (bn1_channel2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (pool1_channel2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  (conv2_channel2): Conv1d(16, 32, kernel_size=(3,), stride=(1,), padding=(1,))  
  (bn2_channel2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (pool2_channel2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  (fc1): Linear(in_features=9600, out_features=128, bias=True)  
  (bn_fc): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (dropout): Dropout(p=0.5, inplace=False)  
  (fc2): Linear(in_features=128, out_features=2, bias=True)  
)
```

| Layer (type)      | Output Shape     | Param #    |
|-------------------|------------------|------------|
| Conv1d-1          | [-1, 16, 600]    | 64         |
| BatchNorm1d-2     | [-1, 16, 600]    | 32         |
| MaxPool1d-3       | [-1, 16, 300]    | 0          |
| Conv1d-4          | [-1, 32, 300]    | 1,568      |
| BatchNorm1d-5     | [-1, 32, 300]    | 64         |
| MaxPool1d-6       | [-1, 32, 150]    | 0          |
| Conv1d-7          | [-1, 16, 600]    | 64         |
| BatchNorm1d-8     | [-1, 16, 600]    | 32         |
| MaxPool1d-9       | [-1, 16, 300]    | 0          |
| Conv1d-10         | [-1, 32, 300]    | 1,568      |
| BatchNorm1d-11    | [-1, 32, 300]    | 64         |
| MaxPool1d-12      | [-1, 32, 150]    | 0          |
| Linear-13         | [-1, 128]        | 1,228,928  |
| BatchNorm1d-14    | [-1, 128]        | 256        |
| Dropout-15        | [-1, 128]        | 0          |
| Linear-16         | [-1, 2]          | 258        |

Total params: 1,232,898  
Trainable params: 1,232,898  
Non-trainable params: 0  

Input size (MB): 1.37  
Forward/backward pass size (MB): 0.74  
Params size (MB): 4.70  
Estimated Total Size (MB): 6.81  

1. 降低了网络复杂度，总参数下降为1.23M
2. 加入了网络训练最大轮数（100）
3. 设置了网络的目标loss，loss1（温度0.005），loss2（应力0.005），loss_total（0.01）
4. 可视化过程加入了loss_total的绘制

## 20250528

### model试训4 ×

对应commit：

#### 待办任务

1. Tensorboard训练过程可视化
2. Loss函数修改，loss_total计算更新。针对应变loss 0.06级别水准仍需调整权重或者更新label训练方法
3. 网络结构瘦身
4. 测试训练完成的代码

未完成，由于通感一体项目

## 20250529

### model试训5

对应commit：20250529_1645_commit

#### 待办5

1. Tensorboard训练过程可视化
2. Loss函数修改，loss_total计算更新。针对应变loss 0.06级别水准仍需调整权重或者更新label训练方法
3. 网络结构瘦身
4. 测试训练完成的代码

修改了batchsize为256
