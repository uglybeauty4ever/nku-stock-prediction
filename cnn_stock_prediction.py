import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import time
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
lr = 1e-4
batch_size = 64
test_batch_size = 250
n_epoch = 400  # 训练的epoch次数修改为400
save_epoch = 100  # 每100个epoch保存一次模型
log_step = 50
train_frac = 0.8  # 训练集比例
test_frac = 0.2  # 测试集比例

# 数据文件位置
data_file = 'merged_file.csv'


# 读取并处理数据
def load_data(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 只选择需要的列
    data = data[['high', 'low', 'close', 'volume', 'open']]  # 只取需要的列

    # 转换数据类型为 float
    features = data[['high', 'low', 'close', 'volume', 'open']].values.astype(np.float32)
    target = data['open'].values.astype(np.float32)  # 使用'open'列作为目标值

    # 创建时间窗口数据（10天为一个时间步长）
    def create_time_windows(data, window_size=10):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size, :-1])  # 提取特征
            y.append(data[i + window_size, -1])  # 提取目标（下一天的开盘价）
        return np.array(X), np.array(y)

    # 将数据转换为时间窗口形式
    X, y = create_time_windows(features)

    # 将数据拆分为训练集和测试集
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 需要调整数据形状，添加一个通道维度
    X_train = np.expand_dims(X_train, axis=1)  # 添加通道维度 -> [samples, 1, 10, 5]
    X_test = np.expand_dims(X_test, axis=1)    # 添加通道维度 -> [samples, 1, 10, 5]

    return X_train, y_train, X_test, y_test

# 定义StockDataset类
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return len(self.target)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 2), stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 2), stride=1, padding=1),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 2), stride=1, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 1), stride=1, padding=1),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=1),
            nn.ReLU())

        self.flatten = nn.Flatten()

        # **修改 `in_features`，确保匹配 Flatten 之后的维度**
        self.fc1 = nn.Linear(3520, 100)
        self.fc2 = nn.Linear(100, 80)
        self.fc3 = nn.Linear(80, 1)  # 输出开盘价（回归任务）

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)  # 输出一个值：开盘价
        return out


# 定义训练和测试的过程
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)  # 回归任务，输出一个标量
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            total_loss += loss.item()

    return total_loss / len(test_loader)


# 评估并生成预测图像
def eval_plot(model, test_loader):
    model.eval()
    preds = []
    labels = []
    loader = tqdm.tqdm(test_loader)

    with torch.no_grad():
        for idx, (data, label) in enumerate(loader):
            data = data.to(device)
            output = model(data).squeeze(1)

            preds += output.cpu().numpy().tolist()
            labels += label.cpu().numpy().tolist()

    # 不进行标准化反操作
    preds = np.array(preds)
    labels = np.array(labels)

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig, ax = plt.subplots(figsize=(12, 6))
    data_x = list(range(len(preds)))
    ax.plot(data_x, preds, label='预测值', color='red')
    ax.plot(data_x, labels, label='实际值', color='blue')

    ax.set_xlabel('时间步')
    ax.set_ylabel('开盘价')
    ax.set_title('预测开盘价 vs 实际开盘价')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('cnn预测图.png')
    plt.show()


if __name__ == '__main__':
    # 加载数据
    X_train, y_train, X_test, y_test = load_data(data_file)

    # 将数据转换为PyTorch Dataset和DataLoader
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # 初始化CNN模型
    model = CNN().to(device)
    print(model)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)

    # 训练模型
    for epoch in range(n_epoch):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, test_loader, criterion)

        print(f"Epoch {epoch + 1}/{n_epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # 保存模型
        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

    print('Training complete.')

    # 绘制预测图像
    eval_plot(model, test_loader)