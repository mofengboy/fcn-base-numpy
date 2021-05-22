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


def _conv_forward(z, K, b, padding=(0, 0)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :return: conv_z: 卷积结果[N,D,oH,oW]
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    oh, ow = (1 + (height - k1), 1 + (width - k2))  # 输出的高度和宽度

    # 扩维
    padding_z = padding_z[:, :, np.newaxis, :, :]  # 扩维[N,C,1,H,W] 与K [C,D,K1,K2] 可以广播
    conv_z = np.zeros((N, D, oh, ow))

    # 批量卷积
    if k1 * k2 < oh * ow * 10:
        K = K[:, :, :, :, np.newaxis, np.newaxis]
        for c in range(C):
            for i in range(k1):
                for j in range(k2):
                    # [N,1,oh,ow]*[D,1,1] =>[N,D,oh,ow]
                    conv_z += padding_z[:, c, :, i:i + oh, j:j + ow] * K[c, :, i, j]
    else:  # 大卷积核，遍历空间更高效
        # print('大卷积核，遍历空间更高效')
        for c in range(C):
            for h in range(oh):
                for w in range(ow):
                    # [N,1,k1,k2]*[D,k1,k2] =>[N,D,k1,k2] => [N,D]
                    conv_z[:, :, h, w] += np.sum(padding_z[:, c, :, h:h + k1, w:w + k2] * K[c], axis=(2, 3))

    # 增加偏置 [N, D, oh, ow]+[D, 1, 1]
    conv_z += b[:, np.newaxis, np.newaxis]
    return conv_z


def conv_forward(z, K, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: conv_z: 卷积结果[N,D,oH,oW]
    oH = (H+2padding-k1)/strides+1
    oW = (W+2padding-k2)/strides+1
    """
    # 长宽方向步长
    sh, sw = strides
    origin_conv_z = _conv_forward(z, K, b, padding)
    # origin_conv_z = c_conv_forward(z, K, b, padding)  # 使用cython
    # 步长为1时的输出卷积尺寸
    N, D, oh, ow = origin_conv_z.shape
    if sh * sw == 1:
        return origin_conv_z
    # 高度方向步长大于1
    elif sw == 1:
        conv_z = np.zeros((N, D, oh // sh, ow))
        for i in range(oh // sh):
            conv_z[:, :, i, :] = origin_conv_z[:, :, i * sh, :]
        return conv_z
    # 宽度方向步长大于1
    elif sh == 1:
        conv_z = np.zeros((N, D, oh, ow // sw))
        for j in range(ow // sw):
            conv_z[:, :, :, j] = origin_conv_z[:, :, :, j * sw]
        return conv_z
    # 高度宽度方向步长都大于1
    else:
        conv_z = np.zeros((N, D, oh // sh, ow // sw))
        for i in range(oh // sh):
            for j in range(ow // sw):
                conv_z[:, :, i, j] = origin_conv_z[:, :, i * sh, j * sw]
        return conv_z


def _insert_zeros(dz, strides):
    """
    想多维数组最后两位，每个行列之间增加指定的个数的零填充
    :param dz: (N,D,H,W),H,W为卷积输出层的高度和宽度
    :param strides: 步长
    :return:
    """
    _, _, H, W = dz.shape
    pz = dz
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pz = np.insert(pz, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pz = np.insert(pz, w, 0, axis=3)
    return pz


def _remove_padding(z, padding):
    """
    移除padding
    :param z: (N,C,H,W)
    :param paddings: (p1,p2)
    :return:
    """
    if padding[0] > 0 and padding[1] > 0:
        return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return z[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return z[:, :, :, padding[1]:-padding[1]]
    else:
        return z


def conv_backward(next_dz, K, z, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积层的反向过程
    :param next_dz: 卷积输出层的梯度,(N,D,H,W),H,W为卷积输出层的高度和宽度
    :param K: 当前层卷积核，(C,D,k1,k2)
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param padding: padding
    :param strides: 步长
    :return:
    """
    N, C, H, W = z.shape
    C, D, k1, k2 = K.shape

    # 卷积核梯度
    # dK = np.zeros((C, D, k1, k2))
    padding_next_dz = _insert_zeros(next_dz, strides)

    # 卷积核高度和宽度翻转180度
    flip_K = np.flip(K, (2, 3))
    # 交换C,D为D,C；D变为输入通道数了，C变为输出通道数了
    swap_flip_K = np.swapaxes(flip_K, 0, 1)
    # 增加高度和宽度0填充
    ppadding_next_dz = np.lib.pad(padding_next_dz, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant',
                                  constant_values=0)
    dz = conv_forward(ppadding_next_dz,
                      swap_flip_K,
                      np.zeros((C,), dtype=np.float))

    # 求卷积和的梯度dK
    swap_z = np.swapaxes(z, 0, 1)  # 变为(C,N,H,W)与
    dK = conv_forward(swap_z, padding_next_dz, np.zeros((D,), dtype=np.float))

    # 偏置的梯度,[N,D,H,W]=>[D]
    db = np.sum(next_dz, axis=(0, 2, 3))  # 在高度、宽度上相加；批量大小上相加

    # 把padding减掉
    dz = _remove_padding(dz, padding)  # dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

    return dK / N, db / N, dz


def max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    pad_h, pad_w = padding
    sh, sw = strides
    kh, kw = pooling
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant',
                           constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * pad_h - kh) // sh + 1
    out_w = (W + 2 * pad_w - kw) // sw + 1

    pool_z = np.zeros((N, C, out_h, out_w), dtype=np.float)

    for i in np.arange(out_h):
        for j in np.arange(out_w):
            pool_z[:, :, i, j] = np.max(padding_z[:, :, sh * i:sh * i + kh, sw * j:sw * j + kw],
                                        axis=(2, 3))
    return pool_z


def max_pooling_backward(next_dz, z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化反向过程
    :param next_dz：损失函数关于最大池化输出的损失
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    pad_h, pad_w = padding
    sh, sw = strides
    kh, kw = pooling
    _, _, out_h, out_w = next_dz.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)
    zeros = np.zeros((N, C, sh, sw))
    for i in np.arange(out_h):
        for j in np.arange(out_w):
            # 找到最大值的那个元素坐标，将梯度传给这个坐标
            cur_padding_z = padding_z[:, :, sh * i:sh * i + kh, sw * j:sw * j + kw]
            cur_padding_dz = padding_dz[:, :, sh * i:sh * i + kh, sw * j:sw * j + kw]
            max_val = np.max(cur_padding_z, axis=(2, 3))  # [N,C]
            cur_padding_dz += np.where(cur_padding_z == max_val[:, :, np.newaxis, np.newaxis],
                                       next_dz[:, :, i:i + 1, j:j + 1],
                                       zeros)
    # 返回时剔除零填充
    return _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

