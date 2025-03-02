import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
lr = 1e-4
batch_size = 64
test_batch_size = 250
n_epoch = 100
save_epoch = 100
train_frac = 0.8
time_step = 10  # 与Transformer代码一致的时间步长

# 数据文件位置
data_file = 'merged_file.csv'  # 修改为您的文件路径


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, T=time_step, train_flag=True):
        data_pd = pd.read_csv(file_path)
        feature_data = data_pd[["high", "low", "open", "close", "volume"]]

        self.train_flag = train_flag
        self.data_train_ratio = 0.9
        self.T = T

        # 标准化处理（与Transformer代码完全一致）
        data_all = np.array(feature_data, dtype=np.float32)
        self.mean = np.mean(data_all, axis=0)
        self.std = np.std(data_all, axis=0)
        data_all = (data_all - self.mean) / self.std

        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_all))
            self.data = data_all[: self.data_len]
        else:
            self.data_len = int((1 - self.data_train_ratio) * len(data_all))
            self.data = data_all[-self.data_len:]

        # 创建时间窗口
        X, y = [], []
        for i in range(len(self.data) - self.T):
            X.append(self.data[i:i + self.T])
            y.append(self.data[i + self.T, 2])  # 使用open列作为目标
        self.features = np.array(X)
        self.targets = np.array(y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def inverse_normalize(self, standardized_data):
        return standardized_data * self.std[2] + self.mean[2]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 2), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 2), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.flatten = nn.Flatten()
        self._init_flatten_size()

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def _init_flatten_size(self):
        x = torch.randn(64, 1, 10, 5)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        self._to_linear = out.view(out.size(0), -1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度 [batch, 1, time_step, features]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


def l2_loss(pred, label):
    return F.mse_loss(pred.squeeze(), label, reduction="mean")


def test(model, test_loader, dataset):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()

            # 限制长度为45，确保与Transformer一致
            outputs = outputs[:46]
            label = label[:46]

            loss = l2_loss(outputs, label.to(device))
            total_loss += loss.item()

            preds += outputs.cpu().numpy().tolist()
            labels += label.cpu().numpy().tolist()

    # 计算准确率（与Transformer代码一致）
    accuracy = 0
    if len(preds) > 1 and len(labels) > 1:
        preds_t = torch.Tensor(preds)
        labels_t = torch.Tensor(labels)
        pred_ = preds_t[1:] > preds_t[:-1]
        label_ = labels_t[1:] > labels_t[:-1]
        accuracy = (label_ == pred_).sum().item() / len(pred_)

    # 反标准化
    inverse_preds = [dataset.inverse_normalize(p) for p in preds]
    inverse_labels = [dataset.inverse_normalize(l) for l in labels]

    return total_loss / len(test_loader), accuracy, inverse_preds, inverse_labels


def eval_plot(preds, labels):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6))

    # 限制X轴为前40个时间步
    data_x = list(range(min(len(preds), 46)))
    preds = preds[:46]
    labels = labels[:46]

    ax.plot(data_x, preds, label='预测值', color='red')
    ax.plot(data_x, labels, label='实际值', color='blue')
    ax.set_xlabel('时间步')
    ax.set_ylabel('开盘价')
    ax.set_title('CNN预测开盘价 vs 实际开盘价')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('cnn_prediction.png')
    plt.show()


if __name__ == '__main__':
    # 初始化数据集
    train_dataset = StockDataset(data_file, train_flag=True)
    test_dataset = StockDataset(data_file, train_flag=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()

    for epoch in range(n_epoch):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 测试阶段
        test_loss, accuracy, preds, labels = test(model, test_loader, test_dataset)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{n_epoch} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:.2%}")

        # 绘制预测图
        if (epoch + 1) % 100 == 0 or epoch == 0:
            eval_plot(preds, labels)

    print('Training complete.')
    torch.save(model.state_dict(), "final_cnn_model.pth")