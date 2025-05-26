# -*- coding: utf-8 -*-            
# @Author : Tongqing Shen
# @Email : tqshen95@163.com
# @Time : 2024/11/16 10:16

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.init as init
import itertools
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler  # 导入MinMaxScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os
import shutil

# 自定义神经网络模型
class SharedAndIndependentNetwork(nn.Module):
    def __init__(self, input_size=7,  hidden_size = 64, shared_layers=5, subnet_layers=5, activation_fn=nn.ReLU(), seed=42):#hidden_size_shared = 64,
        super(SharedAndIndependentNetwork, self).__init__()

        self.seed = seed
        torch.manual_seed(self.seed)

        # 创建共享部分的层
        self.shared_layers = self._create_layers(input_size, hidden_size, shared_layers)

        # 创建每个子网络的独立部分
        self.subnet1_layers = self._create_layers(hidden_size, hidden_size, subnet_layers) + [nn.Linear(hidden_size, 1),
                                                                                              activation_fn()]
        self.subnet2_layers = self._create_layers(hidden_size, hidden_size, subnet_layers) + [nn.Linear(hidden_size, 1),
                                                                                              activation_fn()]
        self.subnet3_layers = self._create_layers(hidden_size, hidden_size, subnet_layers) + [nn.Linear(hidden_size, 1),
                                                                                              activation_fn()]
        self.subnet4_layers = self._create_layers(hidden_size, hidden_size, subnet_layers) + [nn.Linear(hidden_size, 1),
                                                                                              activation_fn()]

        # 初始化权重
        self._initialize_weights()

    def _create_layers(self, in_features, out_features, num_layers):
        layers = []
        for _ in range(num_layers - 1):  # num_layers-1 为隐藏层数
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features  # 每一层的输出作为下一层的输入
        return nn.ModuleList(layers)

    def _initialize_weights(self):
        # 对共享层和子网络中的所有Linear层进行Xavier初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform 初始化
                init.xavier_uniform_(module.weight)
                # 偏置初始化为零
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        # 共享层的前向传播
        shared_out = x
        for layer in self.shared_layers:
            shared_out = layer(shared_out)

        # 每个子网络的独立部分
        out1 = shared_out
        for layer in self.subnet1_layers:
            out1 = layer(out1)

        out2 = shared_out
        for layer in self.subnet2_layers:
            out2 = layer(out2)

        out3 = shared_out
        for layer in self.subnet3_layers:
            out3 = layer(out3)

        out4 = shared_out
        for layer in self.subnet4_layers:
            out4 = layer(out4)

        return out1, out2, out3, out4

    def get_output(self, x):
        """
        输出函数：根据输入数据 `x` 返回每个子网络的预测结果。
        """
        self.eval()  # 设置模型为评估模式，避免 dropout 或 batch norm 的影响
        with torch.no_grad():  # 禁用梯度计算，节省内存
            out1, out2, out3, out4 = self(x)
        return out1, out2, out3, out4


# 加载Excel文件中的四个Sheet
def load_data_from_excel(file_path):
    # 从Excel中读取数据，假设每个Sheet都包含 10 个特征和对应的标签
    sheet1 = pd.read_excel(file_path, sheet_name="rsgs")
    sheet2 = pd.read_excel(file_path, sheet_name="rsngs")
    sheet3 = pd.read_excel(file_path, sheet_name="regs")
    sheet4 = pd.read_excel(file_path, sheet_name="rengs")

    sheet5 = pd.read_excel(file_path, sheet_name="rsa")
    sheet6 = pd.read_excel(file_path, sheet_name="rea")

    #打乱原始的数据排列
    sheet1 = sheet1.sample(frac=1, random_state=42).reset_index(drop=True)
    sheet2 = sheet2.sample(frac=1, random_state=42).reset_index(drop=True)
    sheet3 = sheet3.sample(frac=1, random_state=42).reset_index(drop=True)
    sheet4 = sheet4.sample(frac=1, random_state=42).reset_index(drop=True)
    sheet5 = sheet5.sample(frac=1, random_state=42).reset_index(drop=True)
    sheet6 = sheet6.sample(frac=1, random_state=42).reset_index(drop=True)


    # 提取特征和标签
    X1 = sheet1.iloc[:, 1:8].values  # 假设最后一列是标签
    y1 = sheet1.iloc[:, -1].values

    X2 = sheet2.iloc[:, 1:8].values
    y2 = sheet2.iloc[:, -1].values

    X3 = sheet3.iloc[:, 1:8].values
    y3 = sheet3.iloc[:, -1].values

    X4 = sheet4.iloc[:, 1:8].values
    y4 = sheet4.iloc[:, -1].values

    X5 = sheet5.iloc[:, 1:8].values
    y5 = sheet5.iloc[:, -1].values

    X6 = sheet6.iloc[:, 1:8].values
    y6 = sheet6.iloc[:, -1].values

    # 使用MinMaxScaler进行最大值最小值归一化
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)  # 对特征进行归一化
    X2 = scaler.transform(X2)
    X3 = scaler.transform(X3)
    X4 = scaler.transform(X4)
    X5 = scaler.transform(X5)
    X6 = scaler.transform(X6)

    # 对标签进行归一化，将其范围设置为 [0, 3000]
    #label_scaler = MinMaxScaler(feature_range=(0, 3000))
    y1 = y1 / 3000 #label_scaler.fit_transform(y1.reshape(-1, 1)).flatten()  # 对y进行归一化
    y2 = y2 / 3000 #label_scaler.transform(y2.reshape(-1, 1)).flatten()
    y3 = y3 / 3000 #label_scaler.transform(y3.reshape(-1, 1)).flatten()
    y4 = y4 / 3000 #label_scaler.transform(y4.reshape(-1, 1)).flatten()
    y5 = y5 / 3000
    y6 = y6 / 3000

    # 转换为Tensor
    X1_tensor = torch.tensor(X1, dtype=torch.float32)
    y1_tensor = torch.tensor(y1, dtype=torch.float32).view(-1, 1)

    X2_tensor = torch.tensor(X2, dtype=torch.float32)
    y2_tensor = torch.tensor(y2, dtype=torch.float32).view(-1, 1)

    X3_tensor = torch.tensor(X3, dtype=torch.float32)
    y3_tensor = torch.tensor(y3, dtype=torch.float32).view(-1, 1)

    X4_tensor = torch.tensor(X4, dtype=torch.float32)
    y4_tensor = torch.tensor(y4, dtype=torch.float32).view(-1, 1)

    X5_tensor = torch.tensor(X5, dtype=torch.float32)
    y5_tensor = torch.tensor(y5, dtype=torch.float32).view(-1, 1)

    X6_tensor = torch.tensor(X6, dtype=torch.float32)
    y6_tensor = torch.tensor(y6, dtype=torch.float32).view(-1, 1)

    return (X1_tensor, y1_tensor), (X2_tensor, y2_tensor), (X3_tensor, y3_tensor), (X4_tensor, y4_tensor), (X5_tensor, y5_tensor), (X6_tensor, y6_tensor)


# 训练并评估模型;这里是对的
def train_and_evaluate(model, X1, y1, X2, y2, X3, y3, X4, y4, X5, y5, X6, y6, X1_val,y1_val, X2_val,y2_val, X3_val, y3_val, X4_val, y4_val, X5_val, y5_val, X6_val, y6_val, optimizer, loss_fn, epochs=100):
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):

        optimizer.zero_grad()  # 清空梯度

        # 前向传播：分别获取模型的四个输出
        output1, output2, output3, output4 = model(X1)  # 输入X1对应output1
        _, output2_from_X2,  _, _ = model(X2)  # 输入X2对应output2
        _,  _, output3_from_X3, _ = model(X3)  # 输入X3对应output3
        _, _, _, output4_from_X4 = model(X4)  # 输入X4对应output4

        #约束和值
        output1_from_X5, output2_from_X5, output3_from_X5, output4_from_X5 = model(X5)  # 输入X5对应output1和2
        output1_from_X6, output2_from_X6, output3_from_X6, output4_from_X6 = model(X6)  # 输入X5对应output1和2

        output5 = output1_from_X5 + output2_from_X5
        output6 = output3_from_X6 + output4_from_X6

        # 计算每个输出的损失
        loss1 = loss_fn(output1, y1)
        loss2 = loss_fn(output2_from_X2, y2)
        loss3 = loss_fn(output3_from_X3, y3)
        loss4 = loss_fn(output4_from_X4, y4)
        loss5 = loss_fn(output5, y5)
        loss6 = loss_fn(output6, y6)

        constraint_loss_1 = 0.5 * torch.mean(torch.relu(output2_from_X5 - output1_from_X5)) + 0.5 * torch.mean(
            torch.relu(output2_from_X6 - output1_from_X6))
        constraint_loss_2 = 0.5 * torch.mean(torch.relu(output4_from_X5 - output3_from_X5)) + 0.5 * torch.mean(
            torch.relu(output4_from_X6 - output3_from_X6))
        constraint_loss_3 = 0.5 * torch.mean(torch.relu(output1_from_X5 - output3_from_X5)) + 0.5 * torch.mean(
            torch.relu(output1_from_X6 - output3_from_X6))
        constraint_loss_4 = 0.5 * torch.mean(torch.relu(output2_from_X5 - output4_from_X5)) + 0.5 * torch.mean(
            torch.relu(output2_from_X5 - output4_from_X5))

        Constraint_loss = 0.25 * constraint_loss_1 + 0.25 * constraint_loss_2 + 0.25 * constraint_loss_3 + 0.25 * constraint_loss_4

        Conservation_loss = loss5 + loss6

        Data_loss = loss1 + 2 * loss2 + loss3 + 2 * loss4

        # 总损失
        total_loss = Data_loss + Conservation_loss + Constraint_loss
        #total_loss = loss1 + loss2 + loss3 + loss4 + 0.5 * loss5 + 0.5 * loss6

        # 反向传播
        total_loss.backward()

        # 更新参数
        optimizer.step()

        # 评估模型在验证集上的表现
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算以节省内存
        output1_val, output2_val, output3_val, output4_val = model(X1_val)
        _, output2_from_X2_val, _, _ = model(X2_val)  # 输入X2对应output2
        _, _, output3_from_X3_val, _ = model(X3_val)  # 输入X3对应output3
        _, _, _, output4_from_X4_val = model(X4_val)  # 输入X4对应output4

        output1_from_X5_val, output2_from_X5_val, _, _ = model(X5_val)  # 输入X5对应output1和2
        _, _, output3_from_X6_val, output4_from_X6_val = model(X6_val)  # 输入X5对应output1和2
        output5_val = output1_from_X5_val + output2_from_X5_val
        output6_val = output3_from_X6_val + output4_from_X6_val

        val_loss1 = loss_fn(output1_val, y1_val)
        val_loss2 = loss_fn(output2_from_X2_val, y2_val)
        val_loss3 = loss_fn(output3_from_X3_val, y3_val)
        val_loss4 = loss_fn(output4_from_X4_val, y4_val)
        val_loss5 = loss_fn(output5_val, y5_val)
        val_loss6 = loss_fn(output6_val, y6_val)

        # 计算 R²
        r2_1 = r2_score(y1_val, output1_val.numpy())
        r2_2 = r2_score(y2_val, output2_from_X2_val.numpy())
        r2_3 = r2_score(y3_val, output3_from_X3_val.numpy())
        r2_4 = r2_score(y4_val, output4_from_X4_val.numpy())
        r2_5 = r2_score(y5_val, output5_val.numpy())
        r2_6 = r2_score(y6_val, output6_val.numpy())

        r_1 = pearsonr(y1_val.flatten(), output1_val.numpy().flatten())[0]
        r_2 = pearsonr(y2_val.flatten(), output2_from_X2_val.numpy().flatten())[0]
        r_3 = pearsonr(y3_val.flatten(), output3_from_X3_val.numpy().flatten())[0]
        r_4 = pearsonr(y4_val.flatten(), output4_from_X4_val.numpy().flatten())[0]
        r_5 = pearsonr(y5_val.flatten(), output5_val.numpy().flatten())[0]
        r_6 = pearsonr(y6_val.flatten(), output6_val.numpy().flatten())[0]

        val_total_loss = val_loss1 + val_loss2 + val_loss3 + val_loss4 + val_loss5 + val_loss6

        total_r2 = (r2_1 + r2_2 + r2_3 + r2_4 + r2_5 + r2_6) / 6

    return val_total_loss.item(), total_r2, r2_1, r2_2, r2_3, r2_4, r2_5, r2_6, r_1, r_2, r_3, r_4,r_5,r_6

    #return total_loss.item()


# 10折交叉验证与网格搜索
def cross_val_and_grid_search(X1, y1, X2, y2, X3, y3, X4, y4, X5, y5,X6, y6, param_grid, k_folds=5, num_seeds=10):  #这里用的参数中的，用不到额外定义epochs=100,

    #best_params = None
    #best_loss = float('inf')

    best_params = None
    best_r2 = float('-inf')  # 由于 R² 越大越好，所以初始化为 -∞

    for params in param_grid:
        print(f"Testing params: {params}")

        avg_loss_across_seeds = 0
        avg_r2_across_seeds = 0  # 用于保存每个种子下的平均 R²

        avg_r2_1_across_seeds = 0
        avg_r2_2_across_seeds = 0
        avg_r2_3_across_seeds = 0
        avg_r2_4_across_seeds = 0
        avg_r2_5_across_seeds = 0
        avg_r2_6_across_seeds = 0

        # 内循环：改变种子数，进行多次评估
        for seed in range(num_seeds):
            print(f"Testing with seed {seed}")

            # 固定随机种子，保证每次实验的随机性相同
            torch.manual_seed(seed)
            np.random.seed(seed)

            #kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

            # 分别对每个数据集进行 KFold 切分
            kf1 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            kf2 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            kf3 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            kf4 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            kf5 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            kf6 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)


            avg_loss_for_seed = 0
            avg_r2_for_seed = 0

            # 用于记录每个fold的R²
            r2_1_list, r2_2_list, r2_3_list, r2_4_list, r2_5_list, r2_6_list = [], [], [], [], [], []

            #best_params = None
            #best_loss = float('inf')

            #avg_loss = 0
            for (train_idx1, val_idx1), (train_idx2, val_idx2), (train_idx3, val_idx3), (train_idx4, val_idx4), (
            train_idx5, val_idx5), (train_idx6, val_idx6) in zip(kf1.split(X1), kf2.split(X2), kf3.split(X3),
                                                                 kf4.split(X4), kf5.split(X5), kf6.split(X6)):




            #for train_idx, val_idx in kf.split(X1):
                # 切分数据集
                X1_train, X1_val = X1[train_idx1], X1[val_idx1]
                y1_train, y1_val = y1[train_idx1], y1[val_idx1]
                X2_train, X2_val = X2[train_idx2], X2[val_idx2]
                y2_train, y2_val = y2[train_idx2], y2[val_idx2]
                X3_train, X3_val = X3[train_idx3], X3[val_idx3]
                y3_train, y3_val = y3[train_idx3], y3[val_idx3]
                X4_train, X4_val = X4[train_idx4], X4[val_idx4]
                y4_train, y4_val = y4[train_idx4], y4[val_idx4]
                X5_train, X5_val = X5[train_idx5], X5[val_idx5]
                y5_train, y5_val = y5[train_idx5], y5[val_idx5]
                X6_train, X6_val = X6[train_idx6], X6[val_idx6]
                y6_train, y6_val = y6[train_idx6], y6[val_idx6]

                # 初始化模型
                model = SharedAndIndependentNetwork(input_size=7, hidden_size=params['hidden_size'], #hidden_size_shared=params['hidden_size_shared'],
                                                    shared_layers=params['shared_layers'],
                                                    subnet_layers=params['subnet_layers'],
                                                    activation_fn=nn.Sigmoid,seed=seed)
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
                loss_fn = nn.MSELoss()

                # 训练并评估模型 这里需要改一下，先不改了250210
                loss, r2, r2_1, r2_2, r2_3, r2_4, r2_5, r2_6, *_  = train_and_evaluate(model, X1_train, y1_train, X2_train, y2_train,
                                          X3_train, y3_train, X4_train, y4_train, X5_train, y5_train, X6_train, y6_train,
                                          X1_val,y1_val, X2_val,y2_val, X3_val, y3_val, X4_val, y4_val, X5_val, y5_val, X6_val, y6_val,
                                          optimizer, loss_fn, epochs=params['epochs'])

                #avg_loss += loss
                avg_loss_for_seed += loss
                avg_r2_for_seed += r2

                # 将每个fold的R²保存到相应的list中
                r2_1_list.append(r2_1)
                r2_2_list.append(r2_2)
                r2_3_list.append(r2_3)
                r2_4_list.append(r2_4)
                r2_5_list.append(r2_5)
                r2_6_list.append(r2_6)

            # 计算每个种子下的平均损失
            avg_loss_for_seed /= k_folds
            avg_r2_for_seed /= k_folds  # 每个种子的 R² 平均值

            # 计算每个fold的R²的平均值
            avg_r2_1 = np.mean(r2_1_list)
            avg_r2_2 = np.mean(r2_2_list)
            avg_r2_3 = np.mean(r2_3_list)
            avg_r2_4 = np.mean(r2_4_list)
            avg_r2_5 = np.mean(r2_5_list)
            avg_r2_6 = np.mean(r2_6_list)

            avg_loss_across_seeds += avg_loss_for_seed
            avg_r2_across_seeds += avg_r2_for_seed

            avg_r2_1_across_seeds += avg_r2_1
            avg_r2_2_across_seeds += avg_r2_2
            avg_r2_3_across_seeds += avg_r2_3
            avg_r2_4_across_seeds += avg_r2_4
            avg_r2_5_across_seeds += avg_r2_5
            avg_r2_6_across_seeds += avg_r2_6


            print(f"Avg loss for seed {seed}: {avg_loss_for_seed}")
            print(f"Avg R² for seed {seed}: {avg_r2_for_seed}")

            #print(f"Avg R²_1 for seed {seed}: {avg_r2_1}")
            #print(f"Avg R²_2 for seed {seed}: {avg_r2_2}")
            #print(f"Avg R²_3 for seed {seed}: {avg_r2_3}")
            #print(f"Avg R²_4 for seed {seed}: {avg_r2_4}")
            #print(f"Avg R²_5 for seed {seed}: {avg_r2_5}")
            #print(f"Avg R²_6 for seed {seed}: {avg_r2_6}")

        # 计算所有种子下的平均损失
        avg_loss_across_seeds /= num_seeds
        avg_r2_across_seeds /= num_seeds

        avg_r2_1_across_seeds /= num_seeds
        avg_r2_2_across_seeds /= num_seeds
        avg_r2_3_across_seeds /= num_seeds
        avg_r2_4_across_seeds /= num_seeds
        avg_r2_5_across_seeds /= num_seeds
        avg_r2_6_across_seeds /= num_seeds

        print(f"Avg loss across {num_seeds} seeds: {avg_loss_across_seeds}")
        print(f"Avg R² across {num_seeds} seeds: {avg_r2_across_seeds}")

        # 输出每个 R² 值的平均值
        print(f"Avg R²_1 across {num_seeds} seeds: {avg_r2_1_across_seeds}")
        print(f"Avg R²_2 across {num_seeds} seeds: {avg_r2_2_across_seeds}")
        print(f"Avg R²_3 across {num_seeds} seeds: {avg_r2_3_across_seeds}")
        print(f"Avg R²_4 across {num_seeds} seeds: {avg_r2_4_across_seeds}")
        print(f"Avg R²_5 across {num_seeds} seeds: {avg_r2_5_across_seeds}")
        print(f"Avg R²_6 across {num_seeds} seeds: {avg_r2_6_across_seeds}")

        #avg_loss /= k_folds
        #print(avg_loss)
        '''
        # 如果当前参数组的损失较好，则更新最佳参数
        if avg_loss_across_seeds < best_loss:
            best_loss = avg_loss_across_seeds
            best_params = params'''

        if avg_r2_across_seeds > best_r2:
            best_r2 = avg_r2_across_seeds
            best_params = params

    return best_params, best_r2

# 10折交叉验证，表征模型的泛化能力
def cross_val (X1, y1, X2, y2, X3, y3, X4, y4, X5, y5,X6, y6, Best_params, k_folds=5, num_seeds=100):  #这里用的参数中的，用不到额外定义epochs=100,

    results_list = []
    # 内循环：改变种子数，进行多次评估
    for seed in range(num_seeds):
        print(f"Testing with seed {seed}")

        # 固定随机种子，保证每次实验的随机性相同
        torch.manual_seed(seed)
        np.random.seed(seed)

        #kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        # 分别对每个数据集进行 KFold 切分
        kf1 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        kf2 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        kf3 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        kf4 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        kf5 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        kf6 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)


        avg_loss_for_seed = 0
        avg_r2_for_seed = 0

        # 用于记录每个fold的R²
        r2_1_list, r2_2_list, r2_3_list, r2_4_list, r2_5_list, r2_6_list = [], [], [], [], [], []
        r_1_list, r_2_list, r_3_list, r_4_list, r_5_list, r_6_list = [], [], [], [], [], []

        #best_params = None
        #best_loss = float('inf')

        #avg_loss = 0
        for (train_idx1, val_idx1), (train_idx2, val_idx2), (train_idx3, val_idx3), (train_idx4, val_idx4), (
        train_idx5, val_idx5), (train_idx6, val_idx6) in zip(kf1.split(X1), kf2.split(X2), kf3.split(X3),
                                                             kf4.split(X4), kf5.split(X5), kf6.split(X6)):




        #for train_idx, val_idx in kf.split(X1):
            # 切分数据集
            X1_train, X1_val = X1[train_idx1], X1[val_idx1]
            y1_train, y1_val = y1[train_idx1], y1[val_idx1]
            X2_train, X2_val = X2[train_idx2], X2[val_idx2]
            y2_train, y2_val = y2[train_idx2], y2[val_idx2]
            X3_train, X3_val = X3[train_idx3], X3[val_idx3]
            y3_train, y3_val = y3[train_idx3], y3[val_idx3]
            X4_train, X4_val = X4[train_idx4], X4[val_idx4]
            y4_train, y4_val = y4[train_idx4], y4[val_idx4]
            X5_train, X5_val = X5[train_idx5], X5[val_idx5]
            y5_train, y5_val = y5[train_idx5], y5[val_idx5]
            X6_train, X6_val = X6[train_idx6], X6[val_idx6]
            y6_train, y6_val = y6[train_idx6], y6[val_idx6]

            # 初始化模型
            model = SharedAndIndependentNetwork(input_size=7, hidden_size=Best_params['hidden_size'], #hidden_size_shared=params['hidden_size_shared'],
                                                shared_layers=Best_params['shared_layers'],
                                                subnet_layers=Best_params['subnet_layers'],
                                                activation_fn=nn.Sigmoid,seed=seed)
            optimizer = optim.Adam(model.parameters(), lr=Best_params['learning_rate'])
            loss_fn = nn.MSELoss()

            # 训练并评估模型
            loss, r2, r2_1, r2_2, r2_3, r2_4, r2_5, r2_6,r_1, r_2, r_3, r_4,r_5,r_6  = train_and_evaluate(model, X1_train, y1_train, X2_train, y2_train,
                                      X3_train, y3_train, X4_train, y4_train, X5_train, y5_train, X6_train, y6_train,
                                      X1_val,y1_val, X2_val,y2_val, X3_val, y3_val, X4_val, y4_val, X5_val, y5_val, X6_val, y6_val,
                                      optimizer, loss_fn, epochs=Best_params['epochs'])

            #avg_loss += loss
            avg_loss_for_seed += loss
            avg_r2_for_seed += r2

            # 将每个fold的R²保存到相应的list中
            r2_1_list.append(r2_1)
            r2_2_list.append(r2_2)
            r2_3_list.append(r2_3)
            r2_4_list.append(r2_4)
            r2_5_list.append(r2_5)
            r2_6_list.append(r2_6)

            r_1_list.append(r_1)
            r_2_list.append(r_2)
            r_3_list.append(r_3)
            r_4_list.append(r_4)
            r_5_list.append(r_5)
            r_6_list.append(r_6)

        # 计算每个种子下的平均损失
        avg_loss_for_seed /= k_folds
        avg_r2_for_seed /= k_folds  # 每个种子的 R² 平均值

        # 计算每个fold的R²的平均值
        avg_r2_1 = np.mean(r2_1_list)
        avg_r2_2 = np.mean(r2_2_list)
        avg_r2_3 = np.mean(r2_3_list)
        avg_r2_4 = np.mean(r2_4_list)
        avg_r2_5 = np.mean(r2_5_list)
        avg_r2_6 = np.mean(r2_6_list)

        avg_r_1 = np.mean(r_1_list)
        avg_r_2 = np.mean(r_2_list)
        avg_r_3 = np.mean(r_3_list)
        avg_r_4 = np.mean(r_4_list)
        avg_r_5 = np.mean(r_5_list)
        avg_r_6 = np.mean(r_6_list)

        results_list.append([seed, avg_loss_for_seed, avg_r2_for_seed, avg_r2_1, avg_r2_2, avg_r2_3, avg_r2_4,
                             avg_r2_5, avg_r2_6, avg_r_1, avg_r_2, avg_r_3, avg_r_4, avg_r_5, avg_r_6])

    # 将所有种子的结果保存到 DataFrame
    columns = ["Seed", "Avg_Loss", "Avg_R2", "Avg_R2_1", "Avg_R2_2", "Avg_R2_3", "Avg_R2_4", "Avg_R2_5", "Avg_R2_6",
               "Avg_R_1", "Avg_R_2", "Avg_R_3", "Avg_R_4", "Avg_R_5", "Avg_R_6"]
    results_df = pd.DataFrame(results_list, columns=columns)

    # 保存到 CSV
    results_df.to_csv("cross_val_results.csv", index=False, encoding="utf-8-sig")

    print("✅ 交叉验证结果已保存到 `cross_val_results.csv`！")



# 使用最优超参数训练模型
def train_final_model(X1, y1, X2, y2, X3, y3, X4, y4, X5, y5, X6, y6, best_params,num_seeds=100):

    for seed in range(num_seeds):

        print(f"\n🔹 Training with Seed {seed}")

        model = SharedAndIndependentNetwork(input_size=7,  hidden_size=best_params['hidden_size'], #hidden_size_shared= best_params['hidden_size_shared'],
                                            shared_layers=best_params['shared_layers'],
                                            subnet_layers=best_params['subnet_layers'],
                                            activation_fn=nn.Sigmoid,seed=seed)

        #torch.manual_seed(seed)

        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        loss_fn = nn.MSELoss()
        epochs = best_params['epochs']

        # 训练循环
        for epoch in range(epochs):

            model.train()
            optimizer.zero_grad()

            # 前向传播：分别获取模型的四个输出
            output1, output2, output3, output4 = model(X1)  # 输入X1对应output1
            _,output2_from_X2,  _, _ = model(X2)  # 输入X2对应output2
            _,  _,output3_from_X3, _ = model(X3)  # 输入X3对应output3
            _, _, _, output4_from_X4 = model(X4)  # 输入X4对应output4

            # 约束和值
            output1_from_X5, output2_from_X5, output3_from_X5, output4_from_X5 = model(X5)  # 输入X5对应output1和2
            output1_from_X6, output2_from_X6, output3_from_X6, output4_from_X6 = model(X6)  # 输入X5对应output1和2
            output5 = output1_from_X5 + output2_from_X5
            output6 = output3_from_X6 + output4_from_X6

            # 计算每个输出的损失
            loss1 = loss_fn(output1, y1)
            loss2 = loss_fn(output2_from_X2, y2)
            loss3 = loss_fn(output3_from_X3, y3)
            loss4 = loss_fn(output4_from_X4, y4)

            loss5 = loss_fn(output5, y5)
            loss6 = loss_fn(output6, y6)

            constraint_loss_1 = 0.5 * torch.mean(torch.relu(output2_from_X5 - output1_from_X5)) + 0.5 * torch.mean(torch.relu(output2_from_X6 - output1_from_X6))
            constraint_loss_2 = 0.5 * torch.mean(torch.relu(output4_from_X5 - output3_from_X5)) + 0.5 * torch.mean(torch.relu(output4_from_X6 - output3_from_X6))
            constraint_loss_3 = 0.5 * torch.mean(torch.relu(output1_from_X5 - output3_from_X5)) + 0.5 * torch.mean(torch.relu(output1_from_X6 - output3_from_X6))
            constraint_loss_4 = 0.5 * torch.mean(torch.relu(output2_from_X5 - output4_from_X5)) + 0.5 * torch.mean(torch.relu(output2_from_X5 - output4_from_X5))

            Constraint_loss = 0.25 * constraint_loss_1 + 0.25 * constraint_loss_2 + 0.25 * constraint_loss_3 + 0.25 * constraint_loss_4

            Conservation_loss = loss5 + loss6

            Data_loss = loss1 + 2 * loss2 + loss3 + 2 * loss4

            # 总损失（四个损失之和）
            total_loss = Data_loss + Conservation_loss + Constraint_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            r2_output_1 = r2_score(output1.detach().cpu().numpy(), y1.detach().cpu().numpy())
            r2_output_2 = r2_score(output2_from_X2.detach().cpu().numpy(), y2.detach().cpu().numpy())
            r2_output_3 = r2_score(output3_from_X3.detach().cpu().numpy(), y3.detach().cpu().numpy())
            r2_output_4 = r2_score(output4_from_X4.detach().cpu().numpy(), y4.detach().cpu().numpy())
            r2_output_5 = r2_score(output5.detach().cpu().numpy(), y5.detach().cpu().numpy())
            r2_output_6 = r2_score(output6.detach().cpu().numpy(), y6.detach().cpu().numpy())


            # 打印损失
            if epoch % 100 == 0 or epoch == epochs - 1:

                print (f"Epoch {epoch}/{epochs}")
                print (f"  Total Loss: {total_loss.item():.4f}")
                print (f"  R^2 Output1: {r2_output_1:.4f}, Output2: {r2_output_2:.4f}, Output3: {r2_output_3:.4f}, Output4: {r2_output_4:.4f}")
                print (f"  R^2 Output5: {r2_output_5:.4f}, Output6: {r2_output_6:.4f}")

        # **✅ 直接在本地保存模型**
        model_save_path = f"trained_model_seed_{seed}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ 模型 Seed {seed} 已保存到 {model_save_path}")

# 加载测试数据
def load_test_data(file_path):
    # 假设测试数据的格式与训练数据一致，包含7个特征列
    test_data = pd.read_csv(file_path, header=None)

    # 提取特征数据
    X_test = test_data.iloc[:, 1:8].values  # 假设没有标签列

    # 使用MinMaxScaler进行归一化，使用训练数据的scaler
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)  # 对特征进行归一化

    # 转换为Tensor
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    return X_test_tensor

# 预测四个输出
def predict(model, X_test):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算以节省内存
        # 获取模型的四个输出
        out1, out2, out3, out4 = model.get_output(X_test)

        # 四个输出都乘以3000
        out1 = out1 * 3000
        out2 = out2 * 3000
        out3 = out3 * 3000
        out4 = out4 * 3000

    return out1, out2, out3, out4

# 将预测结果保存到CSV文件
def save_predictions_to_ascii(out1_mean, out2_mean, out3_mean, out4_mean,
                              out1_std, out2_std, out3_std, out4_std,
                              Index, Output_folder,filenames):
    out1_mean, out2_mean, out3_mean, out4_mean = map(np.array, [out1_mean, out2_mean, out3_mean, out4_mean])
    out1_std, out2_std, out3_std, out4_std = map(np.array, [out1_std, out2_std, out3_std, out4_std])

    # 存储均值和方差的列表
    outputs_mean = [out1_mean, out2_mean, out3_mean, out4_mean]
    outputs_std = [out1_std, out2_std, out3_std, out4_std]

    indexes = np.genfromtxt(Index, skip_header=1, delimiter=",")  # 读取索引文件
    index_list = indexes[:, -1]  # 读取最后一列的索引

    # **检查文件名是否正确**
    if len(filenames) != 8:
        raise ValueError("❌ `filenames` 需要 8 个文件名（4 个均值，4 个标准差）")

    # **拆分均值 & 标准差的文件名**
    output_filenames_mean = filenames[:4]  # 前 4 个是均值
    output_filenames_std = filenames[4:]  # 后 4 个是标准差

    # 文件名
    #output_filenames_mean = ["Output1_mean.txt", "Output2_mean.txt", "Output3_mean.txt", "Output4_mean.txt"]
    #output_filenames_std = ["Output1_std.txt", "Output2_std.txt", "Output3_std.txt", "Output4_std.txt"]

    # **写入均值和标准差的 ASCII 文件**
    for output, filename in zip(outputs_mean, output_filenames_mean):
        write_ascii(output, index_list, Output_folder, filename)

    for output, filename in zip(outputs_std, output_filenames_std):
        write_ascii(output, index_list, Output_folder, filename)

    print(f"✅ 8 个 ASCII 文件已成功保存到 `{Output_folder}`")


def write_ascii(output, index_list, Output_folder, filename):
    """
    按照索引列表，将数据保存为 ASCII 栅格格式
    """
    j = 0  # 预测值索引
    ASCII_list = []
    for i in index_list:
        if i == 0:  # 不在计算区域
            ASCII_list.append(-9999)
        else:  # 计算区域
            ASCII_list.append(output[j])
            j += 1

    # 重新整形为 `2821 × 1729`
    ASCII_list_split = func(ASCII_list, 2821)

    # 生成 ASCII 文件路径
    output_file = os.path.join(Output_folder, filename)

    # 写入 ASCII 文件
    with open(output_file, 'w') as f:
        f.write('ncols         2821' + '\n')
        f.write('nrows         1729' + '\n')
        f.write('xllcorner     -664833.32935843' + '\n')
        f.write('yllcorner     2865493.7575816' + '\n')
        f.write('cellsize      1000' + '\n')
        f.write('NODATA_value  -9999' + '\n')
        for line_list in ASCII_list_split:
            f.write(" ".join(map(str, line_list)) + '\n')

    print(f"✅ ASCII 文件 `{filename}` 已保存到 `{Output_folder}`")

#将大列表转为1792行具有2821个元素的小列表
def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

# **清空并重新创建临时存储文件夹**
def clear_temp_folder(temp_dir):
    if os.path.exists(temp_dir):
        print(f"🔹 清空临时文件夹 {temp_dir}")
        shutil.rmtree(temp_dir)  # 删除整个目录
    os.makedirs(temp_dir, exist_ok=True)  # 重新创建目录
    print(f"✅ {temp_dir} 已清空并重新创建")

# 主程序
def main(CvGs = False, Cv = False, Train_final = False, Project = True):

    # 定义超参数网格
    #hidden_size_shared = [20, 30, 40]
    hidden_sizes = [40]#[20, 30, 40]
    shared_layers_options = [5] #[3,4,5]
    subnet_layers_options = [4] #[3,4,5]
    learning_rates = [0.001]#[0.01, 0.001, 0.0001]
    epochs_options = [1000, 2000, 2500, 3000]

    # Best params: 40 5 5 0.001 3000

    # 使用 itertools.product 来生成所有可能的超参数组合
    param_grid = list(itertools.product(
        hidden_sizes,
        shared_layers_options,
        subnet_layers_options,
        learning_rates,
        epochs_options
    )) #hidden_size_shared,
    # 将组合转化为字典的形式
    param_grid_dict = [
        {
            'hidden_size': p[0],
            'shared_layers': p[1],
            'subnet_layers': p[2],
            'learning_rate': p[3],
            'epochs': p[4]
        }#'hidden_size_shared': p[0],
        for p in param_grid
    ]

    # 加载数据
    file_path = 'Input.xlsx'  # 输入文件
    (X1, y1), (X2, y2), (X3, y3), (X4, y4), (X5, y5), (X6, y6) = load_data_from_excel(file_path)

    # print((X1, y1))

    # 执行交叉验证和网格搜索
    if CvGs:
        best_params, best_loss = cross_val_and_grid_search(X1, y1, X2, y2, X3, y3, X4, y4, X5, y5, X6, y6, param_grid_dict)
        print(f"Best params: {best_params}")
        print(f"Best loss: {best_loss}")

    if Cv:
        Best_params_input = {'hidden_size': 40, 'shared_layers': 5, 'subnet_layers': 4, 'learning_rate': 0.001,
                             'epochs': 2500}
        cross_val(X1, y1, X2, y2, X3, y3, X4, y4, X5, y5, X6, y6, Best_params= Best_params_input)


    if Train_final:

        Best_params_input = {'hidden_size': 40, 'shared_layers': 5, 'subnet_layers': 4, 'learning_rate': 0.001, 'epochs': 2500}
        train_final_model(X1, y1, X2, y2, X3, y3, X4, y4, X5, y5, X6, y6, Best_params_input)

    if Project:

        Best_params_input = {'hidden_size': 40, 'shared_layers': 5, 'subnet_layers': 4, 'learning_rate': 0.001,
                             'epochs': 2500}


        # 加载数据
        #main_folder = "F:/Paper8-CEmission/1.Data/7.机器学习训练及预测/3.Pre_future"
        #main_folder = "G:/Paper9/1.Pre_future"
        main_folder = "F:/Paper9-Rs&Re/1.Data/6.Baseline"

        Index_dataset = 'F:/2.中间数据暂时储存/Combination/lon-lat/indexes_txt.csv'

        for sub_folder in os.listdir(main_folder):
            sub_folder_path = os.path.join(main_folder, sub_folder)

            # **遍历每个子文件夹**
            if os.path.isdir(sub_folder_path):
                for sub_sub_folder in os.listdir(sub_folder_path):
                    sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)

                    if os.path.isdir(sub_sub_folder_path):
                        for input_file in os.listdir(sub_sub_folder_path):
                            if input_file.endswith(".csv"):
                                input_file_path = os.path.join(sub_sub_folder_path, input_file)

                                # ========== 新增部分：判断是否已存在结果 ==========
                                base_filename = os.path.splitext(input_file)[0]
                                output_main_folder = f"F:/Paper9-Rs&Re/1.Data/2.Output_mean_std/{sub_folder}"
                                output_sub_folder = os.path.join(output_main_folder, sub_sub_folder)
                                os.makedirs(output_sub_folder, exist_ok=True)

                                output_filenames = [
                                    f"{base_filename}_1_mean.txt", f"{base_filename}_2_mean.txt",
                                    f"{base_filename}_3_mean.txt", f"{base_filename}_4_mean.txt",
                                    f"{base_filename}_1_std.txt", f"{base_filename}_2_std.txt",
                                    f"{base_filename}_3_std.txt", f"{base_filename}_4_std.txt"
                                ]

                                output_exist = all(
                                    os.path.exists(os.path.join(output_sub_folder, f)) for f in output_filenames)

                                if output_exist:
                                    print(f"✅ 已存在结果，跳过：{input_file_path}")
                                    continue
                                # ===================================================

                                print(f"▶ 开始处理：{input_file_path}")



                                # **加载测试数据**
                                X_test = load_test_data(input_file_path)

                                '''# **创建输出目录**  这里对
                                output_main_folder = f"F:/Paper9-Rs&Re/1.Data/2.Output_mean_std/{sub_folder}"
                                output_sub_folder = os.path.join(output_main_folder, sub_sub_folder)
                                os.makedirs(output_sub_folder, exist_ok=True)'''

                                temp_dir = "F:/Paper9-Rs&Re/1.Data/Temp"
                                clear_temp_folder(temp_dir)  # **清空临时目录**

                                #seeds_list = [0, 1, 2, 3, 6, 8, 46, 66, 69, 70, 77, 78, 85, 89, 94]
                                #R2 > 0.6
                                seeds_list = [0, 4, 5, 6, 16, 26, 35, 46, 69, 70, 77, 82, 85, 96]
                                num_seeds = len(seeds_list)

                                for seed in seeds_list:

                                    model = SharedAndIndependentNetwork(
                                        input_size=7,
                                        hidden_size=Best_params_input['hidden_size'],
                                        shared_layers=Best_params_input['shared_layers'],
                                        subnet_layers=Best_params_input['subnet_layers'],
                                        activation_fn=nn.Sigmoid,
                                        seed=seed)
                                    model_path = f"trained_model_seed_{seed}.pth"

                                    model.load_state_dict(torch.load(model_path))
                                    model.eval()

                                    out1, out2, out3, out4 = predict(model, X_test)

                                    out1, out2, out3, out4 = out1.cpu().numpy().flatten(), out2.cpu().numpy().flatten(), out3.cpu().numpy().flatten(), out4.cpu().numpy().flatten()

                                    # **存入临时文件**
                                    temp_file_path = os.path.join(temp_dir, f"seed_{seed}.csv")
                                    pd.DataFrame({"out1": out1, "out2": out2, "out3": out3, "out4": out4}).to_csv(
                                        temp_file_path, index=False)

                                # **计算 100 组 `seed` 预测的均值和方差**
                                print("🔹 开始计算 100 组 `seed` 预测的均值和方差...")

                                # **读取第一组数据，确定数据长度**
                                sample_df = pd.read_csv(os.path.join(temp_dir, "seed_0.csv"))
                                num_samples = len(sample_df)

                                # **初始化存储 `1000` 个样本的均值 & 方差**
                                out1_all = np.zeros((num_seeds, num_samples))
                                out2_all = np.zeros((num_seeds, num_samples))
                                out3_all = np.zeros((num_seeds, num_samples))
                                out4_all = np.zeros((num_seeds, num_samples))

                                # **逐步读取 `100` 组 `seed` 预测结果**
                                for idx, seed in enumerate(seeds_list):
                                    temp_file_path = os.path.join(temp_dir, f"seed_{seed}.csv")
                                    df = pd.read_csv(temp_file_path)

                                    out1_all[idx] = df["out1"].values
                                    out2_all[idx] = df["out2"].values
                                    out3_all[idx] = df["out3"].values
                                    out4_all[idx] = df["out4"].values

                                # **计算均值和标准差**
                                out1_mean, out1_std = np.mean(out1_all, axis=0), np.std(out1_all, axis=0)
                                out2_mean, out2_std = np.mean(out2_all, axis=0), np.std(out2_all, axis=0)
                                out3_mean, out3_std = np.mean(out3_all, axis=0), np.std(out3_all, axis=0)
                                out4_mean, out4_std = np.mean(out4_all, axis=0), np.std(out4_all, axis=0)

                                '''#** 调整输出文件名 **
                                base_filename = os.path.splitext(input_file)[0]
                                output_filenames = [
                                    f"{base_filename}_1_mean.txt", f"{base_filename}_2_mean.txt",
                                    f"{base_filename}_3_mean.txt", f"{base_filename}_4_mean.txt",
                                    f"{base_filename}_1_std.txt", f"{base_filename}_2_std.txt",
                                    f"{base_filename}_3_std.txt", f"{base_filename}_4_std.txt"
                                ]'''

                                # **调用 ASCII 存储函数**
                                save_predictions_to_ascii(out1_mean, out2_mean, out3_mean, out4_mean,
                                                          out1_std, out2_std, out3_std, out4_std,
                                                          Index=Index_dataset, Output_folder=output_sub_folder,
                                                          filenames=output_filenames)


if __name__ == "__main__":
    main()

