import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn


# 将时间序列按照指定时间片进行划分
def split_sequences(sequences, in_steps, out_steps):
    X, y = list(), list()
    for i in range(len(sequences) - 1):
        # 检查是否越界
        if i + in_steps + out_steps > len(sequences):
            break
        seq_x = sequences[i : i + in_steps, :-1]
        seq_y = sequences[i + in_steps : i + in_steps + out_steps, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# reading data frame ==================================================
df = pd.read_csv("goldETF.csv")

in_cols = ["Open", "Low", "Close"]
out_cols = ["Close", "Low"]

# choose a number of time steps
n_steps_in, n_steps_out = 45, 1

# ==============================================================================
# Preparing Model for 'Low'=======================================================
j = 1
dataset_low = np.empty((df[out_cols[j]].values.shape[0], 0))
for i in range(len(in_cols)):
    dataset_low = np.append(
        dataset_low,
        df[in_cols[i]].values.reshape(df[in_cols[i]].values.shape[0], 1),
        axis=1,
    )

dataset_low = np.append(
    dataset_low,
    df[out_cols[j]].values.reshape(df[out_cols[j]].values.shape[0], 1),
    axis=1,
)

# Scaling dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_all = scaler.fit(dataset_low)
scaled_data = scaler_all.transform(dataset_low)

# convert into input/output
x_train, y_train = split_sequences(scaled_data, n_steps_in, n_steps_out)

train_set_size = int(0.2 * scaled_data.shape[0])
x_test, y_test = split_sequences(
    scaled_data[-train_set_size:-1, :], n_steps_in, n_steps_out
)

# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

y_train.size(), x_train.size()

# Build model
##################################################

input_dim = 3
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 50


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])

        return out


model = LSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
)

loss_fn = torch.nn.MSELoss(size_average=True)

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Train model
##################################################################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state

    # Forward pass
    y_train_pred = model(x_train)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler_all.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler_all.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler_all.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler_all.inverse_transform(y_test.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
print("Train Score: %.2f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
print("Test Score: %.2f RMSE" % (testScore))

# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.plot(y_train_pred)
plt.plot(y_train)
plt.show()

# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.plot(y_test_pred)
plt.plot(y_test)
plt.show()
