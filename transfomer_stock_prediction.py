import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch.autograd import Variable
import argparse
import math
import torch.nn.functional as F

torch.random.manual_seed(0)
np.random.seed(0)

flag = input("是否进行股票市场舆情分析[是]|[否](默认):")
if flag == "是":
    vector_size = 6
else:
    vector_size = 5

# 前10天分析第11天
time_step = 10
# 按照9：1划分数据集
train_ratio = 0.9
# 设置训练轮次
epoch = 5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算适合位置编码的div_term, 保证适用于完整维度
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            # 确保 cos 的填充只占据信使的有效部分
            pe[:, 1::2] = torch.cos(position * div_term[: len(pe[:, 1::2][0])])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransAm(nn.Module):
    def __init__(
        self, feature_size=vector_size, num_layers=2, hidden_size=64, dropout=0.1
    ):
        super(TransAm, self).__init__()
        self.model_type = "Transformer"
        self.hidden_size = hidden_size

        self.src_mask = None
        # 将 feature_size 传递给 PositionalEncoding，以便匹配新的输入维度
        self.pos_encoder = PositionalEncoding(d_model=feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=feature_size, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # 添加线性层，将 Transformer 输出的 feature_size 转换为 hidden_size
        self.fc = nn.Linear(feature_size, hidden_size)

        self.decoder = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)  # 转换特征维度到 hidden_size
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class AttnDecoder(nn.Module):
    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(
            in_features=2 * hidden_size, out_features=code_hidden_size
        )
        self.attn2 = nn.Linear(
            in_features=code_hidden_size, out_features=code_hidden_size
        )
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(
            input_size=vector_size, hidden_size=self.hidden_size, num_layers=1
        )
        # 输入特征维度修改为 6
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(
            in_features=code_hidden_size + hidden_size, out_features=hidden_size
        )
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h, y_seq):
        h_ = h.transpose(0, 1)  # (batch_size, time_step, hidden_size)
        batch_size = h.size(0)
        if y_seq.size(0) != batch_size:
            y_seq = y_seq[:batch_size, :]

        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)

        for t in range(self.T):
            if t < h_.size(0):  # Ensure t is within bounds of h_
                x = torch.cat((d, h_[t, :, :].unsqueeze(0)), 2)
                attn_weights = self.attn3(self.tanh(self.attn2(self.attn1(x))))
                attn_weights = F.softmax(attn_weights, dim=0)

                h1 = attn_weights * h_[t, :, :].unsqueeze(0)
                _, states = self.lstm(y_seq[:, t].unsqueeze(0), (h1, s))

                d = states[0]
                s = states[1]
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), h_[-1, :, :]), dim=1)))
        return y_res

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        return Variable(zero_tensor)


class StockDataset(Dataset):
    def __init__(self, file_path, train_flag):
        # 读取数据
        df = pd.read_csv(file_path)
        # 确定特征数据
        if vector_size == 5:
            data = df[["high", "low", "open", "close", "volume"]]
        else:
            data = df[["high", "low", "open", "close", "volume", "avg_sentiment_score"]]
        # 转原始数据为numpy数组
        data = np.array(data, dtype=np.float32)
        # 获取特征数据平均值
        self.mean = np.mean(data, axis=0)
        # 获取特征数据方差
        self.std = np.std(data, axis=0)
        # 获取标准化后的数据
        self.data = (data - self.mean) / self.std
        # 划分训练集与验证集
        train_size = int(len(self.data) * train_ratio)
        if train_flag:
            self.data = self.data[:train_size]
        else:
            self.data = self.data[train_size:]
        # 确定数据的时间序列长度
        self.time_step = time_step

    def __len__(self):
        # 返回数据长度
        return int(len(self.data)) - self.time_step

    def __getitem__(self, index):
        # 按照时间序列长度划分数据
        data = self.data[index : index + self.time_step]
        label = self.data[index + self.time_step, 3]
        return data, label

    def inverse_normalize(self, standardized_data):
        # 对每个数据进行反标准化
        if isinstance(standardized_data, list):
            return [self.inverse_normalize(x) for x in standardized_data]
        return standardized_data * self.std[3] + self.mean[3]


def l2_loss(pred, label):
    # 修正 size_average 警告，使用 reduction='mean' 代替
    loss = torch.nn.functional.mse_loss(pred, label, reduction="mean")
    return loss


def train_once(encoder, decoder, train_loader, encoder_optim, decoder_optim):
    # 设置为训练模式
    encoder.train()
    decoder.train()
    # 设置进度条
    loader = tqdm.tqdm(train_loader)
    # 初始化平均损失值
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        # 转换数据类型
        data = data.float()
        label = label.float()
        # 交换原始数据0，1维向量
        transposed_data = data.transpose(0, 1)
        # 对数据进行编码
        encode_data = encoder(transposed_data)
        # 对数据进行解码
        decode_data = decoder(encode_data.transpose(0, 1), data)
        # 获取预测值
        pred = decode_data.squeeze(1)
        # 梯度清零
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        # 计算损失值
        loss = l2_loss(pred, label)
        # 进行反向传播，计算梯度
        loss.backward()
        # 更新参数
        encoder_optim.step()
        decoder_optim.step()
        # 累加损失值
        loss_epoch += loss.detach().item()
    loss_epoch /= len(loader)
    return loss_epoch


def eval_once(encoder, decoder, val_loader):
    # 设置为评估模式
    encoder.eval()
    decoder.eval()
    # 取消数据打乱
    val_loader.shuffle = False
    # 设置进度条
    loader = tqdm.tqdm(val_loader)
    # 初始化损失值
    loss_epoch = 0
    # 初始化预测值和原始值
    preds = []
    labels = []
    for idx, (data, label) in enumerate(loader):
        # 转换数据类型
        data = data.float()
        label = label.float()
        # 交换原始数据0，1维向量
        transposed_data = data.transpose(0, 1)
        # 对数据进行编码
        encode_data = encoder(transposed_data)
        # 对数据进行解码
        decode_data = decoder(encode_data.transpose(0, 1), data)
        # 获取预测值
        output = decode_data.squeeze(1)
        # 确定输出时间步长
        min_len = min(output.size(0), label.size(0))
        output = output[:min_len]
        label = label[:min_len]
        # 计算损失值
        loss = l2_loss(output, label)
        loss_epoch += loss.detach().item()
        # 获取预测数据和原始数据
        preds += output.detach().tolist()
        labels += label.detach().tolist()
    # 计算准确率
    if len(preds) > 1 and len(labels) > 1:
        preds = torch.Tensor(preds)
        labels = torch.Tensor(labels)
        # 判断增长趋势
        pred_ = preds[1:] > preds[:-1]
        label_ = labels[1:] > labels[:-1]
        accuracy = (label_ == pred_).sum().item() / len(pred_)
    else:
        accuracy = 0
    loss_epoch /= len(loader)
    return loss_epoch, accuracy, preds, labels


def data_plot(preds, labels, val_data):
    # 设置时间步长
    day = 40
    # x轴数据
    days = list(range(day))
    # y轴数据
    preds = preds[:day]
    labels = labels[:day]
    # 反标准化
    preds = [val_data.inverse_normalize(p) for p in preds]
    labels = [val_data.inverse_normalize(l) for l in labels]
    # 设置画布大小
    fig, ax = plt.subplots(figsize=(12, 6))
    # 添加x,y内容
    ax.plot(days, preds, label="预测值", color="red")
    ax.plot(days, labels, label="实际值", color="blue")
    # 设置横纵主题
    ax.set_xlabel("时间步")
    ax.set_ylabel("开盘价")
    # 设置图表主题
    ax.set_title("Transformer预测开盘价 vs 实际开盘价")
    # 固定y轴范围
    plt.ylim(4000, 5000)
    # 自定义参数
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    # 绘图
    plt.show()
    # 保存
    plt.savefig("result.png")


def main():
    # 初始化数据集
    train_data = StockDataset(file_path="merged_file.csv", train_flag=True)
    val_data = StockDataset(file_path="merged_file.csv", train_flag=False)
    # 初始化编码层解码层
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, drop_last=False)
    # 初始化编码器和解码器
    encoder = TransAm(feature_size=vector_size)
    decoder = AttnDecoder(code_hidden_size=64, hidden_size=64, time_step=time_step)
    # 初始化优化器
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=0.001)
    # 训练阶段
    print("-------------------------STAGE: TRAIN -------------------------")
    for idx in range(epoch):
        loss = train_once(encoder, decoder, train_loader, encoder_optim, decoder_optim)
        print("epoch:{:5d}, loss:{}".format(idx + 1, loss))
    # 验证阶段
    print("-------------------------STAGE: VALIDATION -------------------------")
    loss, accuracy, preds, labels = eval_once(encoder, decoder, val_loader)
    print("loss:{}, accuracy:{}".format(loss, accuracy))
    # 绘图
    data_plot(preds, labels, val_data)


if __name__ == "__main__":
    main()
