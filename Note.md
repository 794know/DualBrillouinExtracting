# Note

## 20250524

### 贵江Loss函数思路  

BGS全部归一化后添加噪声，噪声值用1除  

### 训练思路  

数据以及label全部放入压缩文件，网络1对应某训练集即可读取对应文件夹进行训练。  

## 20250526

### model试训1  

#### 项目结构  

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

#### Loss函数设计 18:47  

loss1，loss2，total_loss 均需要可视化。同时，可能需要根据两个label的scale进行label的归一化。  
学习目标，测试集以及验证集的划分需要进一步设置

发现错误：  
label的归一化应该按照最大值归一化，而不是按照每行归一化！  
明日任务：  
写好labels归一化的函数  

## 20250527

### model试训2

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

### Label归一化函数的构建

label的原始结构为[2*132300]，进入dataloader后会被转置为[132300*2]，这样需要我对两列的最大值归一化  

### 解释器环境配置

更新至python 3.9.21环境，原环境为pytorch。训练时仍然使用实验室电脑pytorch环境。训练epoch数量：30  
训练结果中温度应力loss下降均有一定趋势，但是validation loss比较大
在原本的训练文件中需要添加以下功能：

1. loss下降至所需目标
2. 网络drop训练并且冻结网络参数
3. 确定validation loss的下降方法，可能训练数据信噪比6 dB较低了感觉
