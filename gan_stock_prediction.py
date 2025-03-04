import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# 读取数据
df = pd.read_csv("merged_file.csv")

# 选择特征列 (去掉重复的 'close' 列)
feature_columns = [ 'high', 'low', 'open', 'volume']
target_column = 'close'

# 提取特征（X）和目标变量（y）
X = df[feature_columns].values
y = df[target_column].values

# 归一化数据（Min-Max 归一化）
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
train_size = int(len(X) * 0.8)  # 80% 作为训练数据
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# 转换为 3D 输入格式 (样本数, 时间步, 特征数)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 生成器模型（生成股票价格）
def make_generator_model(input_dim, output_dim, feature_size):
    model = Sequential()
    model.add(GRU(units=256, return_sequences=True, input_shape=(input_dim, feature_size)))
    model.add(GRU(units=128, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))  # 线性激活层用于回归任务
    return model

# 判别器模型（判断生成的股票价格真假）
def make_discriminator_model(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(GRU(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 输出 0~1 概率
    return model

# GAN 类：包含生成器和判别器
class GAN:
    def __init__(self, generator, discriminator, opt):
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=opt["lr"])  # 修改为 learning_rate
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=opt["lr"])  # 修改为 learning_rate
        self.batch_size = opt['bs']

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_y = self.generator(real_x, training=True)

            real_output = self.discriminator(real_y, training=True)
            fake_output = self.discriminator(fake_y, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, X_train, y_train, opt):
        epochs = opt["epoch"]
        for epoch in range(epochs):
            gen_loss, disc_loss = self.train_step(X_train, y_train)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")

# 设置优化参数
opt = {"lr": 0.00016, "epoch": 100, "bs": 128}

# 构建生成器和判别器模型
generator = make_generator_model(1, 1, X_train.shape[2])
discriminator = make_discriminator_model((1, 1))

# 初始化 GAN
gan = GAN(generator, discriminator, opt)

# 训练 GAN 模型
gan.train(X_train, y_train, opt)

# 使用训练好的生成器进行股票价格预测
predicted_prices_scaled = generator.predict(X_test)

# 反归一化预测值
predicted_prices = scaler_y.inverse_transform(predicted_prices_scaled)
actual_prices = scaler_y.inverse_transform(y_test)

# 可视化预测结果
plt.figure(figsize=(16, 8))
plt.plot(actual_prices, label="Real Close Price")
plt.plot(predicted_prices, label="Predicted Close Price", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Stock Price Prediction using GAN")
plt.show()

# 计算均方根误差 RMSE
RMSE = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print('-- Test RMSE -- ', RMSE)