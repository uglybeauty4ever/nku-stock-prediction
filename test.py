import numpy as np
from numpy import array

def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)-1):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix+1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# 示例多变量时间序列数据
sequences = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50],
    [40, 50, 60],
    [50, 60, 70]
])

# 设置输入和输出的时间步数
n_steps_in = 2
n_steps_out = 1

# 调用函数进行数据分割
X, y = split_sequences(sequences, n_steps_in, n_steps_out)

print("输入序列 X:")
print(X)
print("输出序列 y:")
print(y)