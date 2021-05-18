import numpy as np


# 全连接层前向传播
def fc_forward(w, x, b):
    return np.dot(w, x) + b


def fc_backward(next_dz, W, z):
    """
    全连接层的反向传播
    :param next_dz: 下一层的梯度
    :param W: 当前层的权重
    :param z: 当前层的输出
    :return:
    """
    N = z.shape[1]
    dz = np.dot(next_dz.T, W)  # 当前层的梯度
    dw = np.dot(z, next_dz.T).T  # 当前层权重的梯度
    db = np.sum(next_dz, axis=1, keepdims=True)  # 当前层偏置的梯度, N个样本的梯度求和
    return dw / N, db / N, dz
