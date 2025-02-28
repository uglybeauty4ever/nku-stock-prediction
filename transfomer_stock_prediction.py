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

time_step = 10  # 根据10天数据预测第11天

df = pd.read_csv("merged_file.csv")


flag = input("是否进行股票市场舆情分析[是]|[否]:")
if flag == "是":
    vector_size = 6
else:
    vector_size = 5


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
    ):  # 修改特征维度为 6
        super(TransAm, self).__init__()
        self.model_type = "Transformer"
        self.hidden_size = hidden_size

        self.src_mask = None
        # 将 feature_size 传递给 PositionalEncoding，以便匹配新的输入维度
        self.pos_encoder = PositionalEncoding(d_model=feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=1, dropout=dropout, batch_first=True
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
    def __init__(self, file_path, train_ratio, time_step, train_flag):
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
        """
        反标准化方法，支持处理嵌套列表
        :param standardized_data: 标准化后的数据
        :return: 原始数据
        """
        if isinstance(standardized_data, list):
            return [self.inverse_normalize(x) for x in standardized_data]
        return standardized_data * self.std[3] + self.mean[3]


def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, size_average=True)
    return loss


def train_once(encoder, decoder, dataloader, encoder_optim, decoder_optim):
    encoder.train()
    decoder.train()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        data_x = data.transpose(0, 1).float()  # (time_step, batch_size, num_features)
        label = label.float()

        # 打印数据形状进行调试
        print("data_x shape:", data_x.shape)
        print("label shape:", label.shape)

        code_hidden = encoder(data_x)
        code_hidden = code_hidden.transpose(0, 1)
        output = decoder(code_hidden, data.float())

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        loss = l2_loss(output.squeeze(1), label)
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()
        loss_epoch += loss.detach().item()
    loss_epoch /= len(loader)
    return loss_epoch


def l2_loss(pred, label):
    # 修正 size_average 警告，使用 reduction='mean' 代替
    loss = torch.nn.functional.mse_loss(pred, label, reduction="mean")
    return loss


def eval_once(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    preds = []
    labels = []
    for idx, (data, label) in enumerate(loader):
        data_x = data.transpose(0, 1).float()  # (time_step, batch_size, num_features)
        label = label.float()
        data_y = data.float().transpose(0, 1)

        code_hidden = encoder(data_x)
        output = decoder(code_hidden, data_y).squeeze(1)

        # 确保 output 和 label 的长度一致
        min_len = min(output.size(0), label.size(0))
        output = output[:min_len]
        label = label[:min_len]

        loss = l2_loss(output, label)
        loss_epoch += loss.detach().item()

        preds += output.detach().tolist()
        labels += label.detach().tolist()

    if len(preds) > 1 and len(labels) > 1:
        preds = torch.Tensor(preds)
        labels = torch.Tensor(labels)
        pred_ = preds[1:] > preds[:-1]
        label_ = labels[1:] > labels[:-1]
        accuracy = (label_ == pred_).sum().item() / len(pred_)
    else:
        accuracy = 0

    loss_epoch /= len(loader)
    return loss_epoch, accuracy


def eval_plot(encoder, decoder, dataloader, dataset):
    dataloader.shuffle = False
    preds = []
    labels = []
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)
    for idx, (data, label) in enumerate(loader):
        data_x = data.transpose(0, 1).float()
        data_y = data.float()

        # 强制调整 data_y 的时间步长度为 time_step
        if data_y.size(1) > time_step:
            data_y = data_y[:, :time_step, :]
        elif data_y.size(1) < time_step:
            padding = torch.zeros(
                (data_y.size(0), time_step - data_y.size(1), data_y.size(2))
            )
            data_y = torch.cat((data_y, padding), dim=1)

        code_hidden = encoder(data_x)

        # 确保 code_hidden 和 data_y 有一致的 batch size
        if code_hidden.size(0) != data_y.size(0):
            if code_hidden.size(0) > data_y.size(0):
                code_hidden = code_hidden[: data_y.size(0)]
            else:
                data_y = data_y[: code_hidden.size(0)]

        output = decoder(code_hidden, data_y)
        preds += output.detach().tolist()
        labels += label.detach().tolist()

    # 裁剪 preds 和 labels 以匹配较短的长度
    min_len = min(len(preds), len(labels))
    preds = preds[:40]
    labels = labels[:40]

    # 反标准化
    inverse_preds = [dataset.inverse_normalize(p) for p in preds]
    inverse_labels = [dataset.inverse_normalize(l) for l in labels]

    preds = inverse_preds
    labels = inverse_labels

    fig, ax = plt.subplots(figsize=(12, 6))
    data_x = list(range(len(preds)))
    ax.plot(data_x, preds, label="predict", color="red")
    ax.plot(data_x, labels, label="ground truth", color="blue")
    ax.set_xlabel("时间步")
    ax.set_ylabel("开盘价")
    ax.set_title("Transformer预测开盘价 vs 实际开盘价")
    plt.ylim(4000, 5000)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.savefig("shangzheng-tran-lstmnoavg.png")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


def main():
    # 训练集
    dataset_train = StockDataset(
        file_path="merged_file.csv", train_ratio=0.9, time_step=10, train_flag=True
    )
    # 验证集
    dataset_val = StockDataset(
        file_path="merged_file.csv", train_ratio=0.9, time_step=10, train_flag=False
    )

    # 将 drop_last=True 仅用于训练集，确保验证集的批次不会丢失数据
    train_loader = DataLoader(
        dataset_train, batch_size=32, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, drop_last=False)

    # 初始化编码器和解码器
    encoder = TransAm(feature_size=vector_size)  # 设置特征维度为 6
    decoder = AttnDecoder(code_hidden_size=64, hidden_size=64, time_step=time_step)

    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=0.001)

    total_epoch = 10
    for epoch_idx in range(total_epoch):
        train_loss = train_once(
            encoder, decoder, train_loader, encoder_optim, decoder_optim
        )
        print("stage: train, epoch:{:5d}, loss:{}".format(epoch_idx, train_loss))

        if epoch_idx % 9 == 0:
            eval_loss, accuracy = eval_once(encoder, decoder, val_loader)
            print(
                "####stage: test, epoch:{:5d}, loss:{}, accuracy:{}".format(
                    epoch_idx, eval_loss, accuracy
                )
            )
            eval_plot(encoder, decoder, val_loader, dataset_val)


if __name__ == "__main__":
    main()
