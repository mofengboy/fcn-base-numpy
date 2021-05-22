# 卷积神经网络关键点分析

## 核心矩阵
### 卷积层
1. 卷积层矩阵：形状(N,C,H,W)，N为batch_size，C为通道数，H为高度，W为宽度
2. 卷积核：形状(C,D,k1,k2), C为输入通道数，D为输出通道数，k1,k2分别为卷积核的高度和宽度
3. 偏置b: 形状(D,) 
4. 卷积结果为[N,D,oH,oW]
  
  其中oH = (H+2padding-k1)/strides+1 ; oW = (W+2padding-k2)/strides+1
  
**前向传播代码** 

```Python
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
```

**反向传播代码**

```Python
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
```

### 池化层
1. 该层为静态层，仅可以设置超参，没有可以学习的参数。

2. 一般分为最大池化和平均池化，常用最大池化。

3. 常用的参数值为f=2和s=2,这时高度和宽度都会减少一半，通道数不变。

4. 设池化层的输入维度分别为[H,W,C],其中H为高度，W为宽度，C为通道数，则输出维度为[(H-F)/s+1,(W-F)/s+1,C]

**前向传播代码**

```Python
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
```

**反向传播代码**

```Python
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

```

## 参数初始化
这部分仅仅介绍卷积层部分的参数，全连接层的见上篇文章。

采用的是xavier初始化方法
```Python
    self.Parameters["K1"] = np.random.randn(1, 4, 3, 3) / np.sqrt(4 / 2)
    self.Parameters["K2"] = np.random.randn(4, 16, 3, 3) / np.sqrt(16 / 2)
    self.Parameters["Kb1"] = np.zeros(4)
    self.Parameters["Kb2"] = np.zeros(16)
```
但是很奇怪用下面这种初始化方式会报错，我认为这种方式虽然不如上面的方法好，但是也不太应该报错吧，先留个疑问，有时间研究一下，也欢迎大佬指出原因。

报错提示：
```Python
        # self.Parameters["K1"] = np.random.randn(1, 4, 3, 3)
        # self.Parameters["K2"] = np.random.randn(4, 16, 3, 3)
        
        # RuntimeWarning: divide by zero encountered in log
        #   loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
```

# [神经网络底层原理]-使用基于numpy的卷积网络+全连接神经网络识别mnist数据集
## 数据读取部分
### 数据集介绍
> 数据集地址： https://cs.nyu.edu/~roweis/data.html

数据包括从 "0 "到 "9 "的8位灰度图像；每类约6K训练实例；1K测试实例。
### 数据读取函数

ps:由于训练集是按照0-9,每类6K顺序排列，所以我这里将数据读入numpy数组后，对训练集的顺序进行了打乱操作（shuffle）,这是因为固定的数据集顺序，严重限制了梯度优化方向的可选择性，导致收敛点选择空间严重变少，容易导致过拟合。
```Python
    def loadData(self, file_path):
        # 使用scipy读入mat文件数据
        mnist_all = sio.loadmat(file_path)
        train_raw = []
        test_raw = []
        # 依次读入数据集0-9
        for i in range(10):
            train_temp = mnist_all["train" + str(i)]
            for j in train_temp:
                j = np.array(j)
                j = j.reshape((1, 28, 28))
                train_raw.append([j, i])
        for i in range(10):
            test_temp = mnist_all["test" + str(i)]
            for j in test_temp:
                j = np.array(j)
                j = j.reshape((1, 28, 28))
                test_raw.append([j, i])

        Train = np.array(train_raw)
        Test = np.array(test_raw)
        # print(Train[0][0][0])
        # img = Image.fromarray(Train[0][0][0])
        # img.show()
        # 随机打乱数据
        np.random.shuffle(Test)
        np.random.shuffle(Train)
        self.Train = Train
        self.Test = Test

    # 批量获取训练数据集
    def getBatchTrain(self, is_train=True, batch_size=128, offset=0):
        if is_train:
            data = self.Train
        else:
            data = self.Test
        X = []
        Y = []
        y_temp = np.eye(10)
        length = data.shape[0]
        for i in range(batch_size):
            X.append(data[(offset + i) % length][0])
            Y.append(y_temp[data[(offset + i) % length][1]])
        X = np.array(X)
        Y = np.array(Y).T
        # print(X.shape)
        #     归一化
        X = X / 225.0
        return X, Y
```

### 参数初始化部分

包含两部分，卷积层的参数和全连接层的参数，可以发现卷积层的参数并不多，这是因为卷积层使用了共享参数的思想。

```Python
    def initParam(self, fc_init_type="xavier"):
        # 初始化filter
        self.Parameters["K1"] = np.random.randn(1, 4, 3, 3) / np.sqrt(4 / 2)
        self.Parameters["K2"] = np.random.randn(4, 16, 3, 3) / np.sqrt(16 / 2)
        self.Parameters["Kb1"] = np.zeros(4)
        self.Parameters["Kb2"] = np.zeros(16)

        # 很奇怪用下面这种初始化方式会报错
        # self.Parameters["K1"] = np.random.randn(1, 4, 3, 3)
        # self.Parameters["K2"] = np.random.randn(4, 16, 3, 3)
        # RuntimeWarning: divide by zero encountered in log
        #   loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数

        # 初始化W
        if fc_init_type == "xavier":
            self.Parameters["W1"] = np.random.randn(256, 400) / np.sqrt(256 / 2)
            self.Parameters["W2"] = np.random.randn(128, 256) / np.sqrt(128 / 2)
            self.Parameters["W3"] = np.random.randn(10, 128) / np.sqrt(10 / 2)
        elif fc_init_type == "normal":
            self.Parameters["W1"] = np.random.normal(loc=0.0, scale=1.0, size=(256, 400)) * 0.01
            self.Parameters["W2"] = np.random.normal(loc=0.0, scale=1.0, size=(128, 256)) * 0.01
            self.Parameters["W3"] = np.random.normal(loc=0.0, scale=1.0, size=(10, 128)) * 0.01
        elif fc_init_type == "rand":
            self.Parameters["W1"] = np.random.rand(256, 400) * 0.01
            self.Parameters["W2"] = np.random.rand(128, 256) * 0.01
            self.Parameters["W3"] = np.random.rand(10, 128) * 0.01
        else:
            raise Exception("W无效的参数初始化类型")

        # 初始化b
        self.Parameters["B1"] = np.zeros((256, 1))
        self.Parameters["B2"] = np.zeros((128, 1))
        self.Parameters["B3"] = np.zeros((10, 1))
        
 ```
 
 ### 前向传播部分和反向传播部分
 
```Python
    def forward(self, train_data):
        self.nodes["Conv1"] = \
            layer.conv_forward(train_data, self.Parameters["K1"], self.Parameters["Kb1"])
        self.nodes["Maxpool1"] = layer.max_pooling_forward(self.nodes["Conv1"], (2, 2), (2, 2))
        self.nodes["Conv2"] = \
            layer.conv_forward(self.nodes["Maxpool1"], self.Parameters["K2"], self.Parameters["Kb2"])
        self.nodes["MaxPool2"] = layer.max_pooling_forward(self.nodes["Conv2"], (2, 2))

        self.nodes["X2"] = self.nodes["MaxPool2"].reshape((128, -1)).T

        self.nodes["A1"] = layer.fc_forward(self.Parameters["W1"], self.nodes["X2"], self.Parameters["B1"])
        self.nodes["Z1"] = activations.relu_forward(self.nodes["A1"])
        self.nodes["A2"] = layer.fc_forward(self.Parameters["W2"], self.nodes["Z1"], self.Parameters["B2"])
        self.nodes["Z2"] = activations.relu_forward(self.nodes["A2"])
        self.nodes["A3"] = layer.fc_forward(self.Parameters["W3"], self.nodes["Z2"], self.Parameters["B3"])
        self.nodes["Z3"] = activations.relu_forward(self.nodes["A3"])
        self.nodes["y"] = np.argmax(self.nodes["A3"], axis=0)
        return self.nodes["y"]

    def backward(self, train_data, y_true):
        loss, self.gradients["A3"] = losses.cross_entropy_loss(self.nodes["A3"], y_true)
        self.gradients["W3"], self.gradients["B3"], self.gradients["Z2"] = \
            layer.fc_backward(self.gradients["A3"], self.Parameters["W3"], self.nodes["Z2"])

        self.gradients["A2"] = activations.relu_backward(self.gradients["Z2"].T, self.nodes["A2"])
        self.gradients["W2"], self.gradients["B2"], self.gradients["Z1"] = \
            layer.fc_backward(self.gradients["A2"], self.Parameters["W2"], self.nodes["Z1"])

        self.gradients["A1"] = activations.relu_backward(self.gradients["Z1"].T, self.nodes["A1"])
        self.gradients["W1"], self.gradients["B1"], self.gradients["Z1"] = \
            layer.fc_backward(self.gradients["A1"], self.Parameters["W1"], self.nodes["X2"])

        self.gradients["Z1"] = self.gradients["Z1"].reshape((128, 16, 5, 5))

        self.gradients["Maxpool2"] = layer.max_pooling_backward(self.gradients["Z1"], self.nodes["Conv2"], (2, 2))
        self.gradients["K2"], self.gradients["Kb2"], self.gradients["KZ2"] = \
            layer.conv_backward(self.gradients["Maxpool2"], self.Parameters["K2"], self.nodes["Maxpool1"])

        self.gradients["Maxpool1"] = \
            layer.max_pooling_backward(self.gradients["KZ2"], self.nodes["Conv1"], (2, 2))
        self.gradients["K1"], self.gradients["Kb1"], self.gradients["KZ1"] = \
            layer.conv_backward(self.gradients["Maxpool1"], self.Parameters["K1"], train_data)

        return loss
```

## 训练过程
### 代码
训练过程中每200次统计并输出损失和当前的准确率，最后输出验证集的正确率并画图。

```Python
    def train(self, epoch, batch_size, lr):
        plt_x = []
        plt_acc = []
        plt_loss = []

        for i in range(epoch * 60000 // batch_size):
            X, Y = self.getBatchTrain(batch_size=batch_size, offset=i * batch_size)

            y_p = self.forward(X)
            loss = self.backward(X, Y)
            y_true = np.argmax(Y, axis=0)
            # print("y_p:")
            # print(y_p)
            # print("y_true:")
            # print(y_true)

            # 参数更新
            self.Parameters["K2"] -= lr * self.gradients["K2"]
            self.Parameters["Kb2"] -= lr * self.gradients["Kb2"]
            self.Parameters["K1"] -= lr * self.gradients["K1"]
            self.Parameters["Kb1"] -= lr * self.gradients["Kb1"]

            self.Parameters["W3"] -= lr * self.gradients["W3"]
            self.Parameters["B3"] -= lr * self.gradients["B3"]
            self.Parameters["W2"] -= lr * self.gradients["W2"]
            self.Parameters["B2"] -= lr * self.gradients["B2"]
            self.Parameters["W1"] -= lr * self.gradients["W1"]
            self.Parameters["B1"] -= lr * self.gradients["B1"]

            if i % 200 == 0:
                y_true = np.argmax(Y, axis=0)
                acc = self.getAcc(X, y_true)
                print("第" + str(i) + "次准确率为：" + str(acc))
                print("第" + str(i) + "次损失为：" + str(loss))
                plt_x.append(i)
                plt_acc.append(acc)
                plt_loss.append(loss)

        plt.xlabel('i')
        plt.ylabel('ACC')
        plt.plot(plt_x, plt_acc, linewidth=2, color='blue', linestyle='--')
        plt.show()
        plt.xlabel('i')
        plt.ylabel('Loss')
        plt.plot(plt_x, plt_loss, linewidth=2, color='red', linestyle='-.')
        plt.show()
```

### 训练结果

>  epoch = 15 

>  lr = 0.01

>  batch_size = 128

![mark](https://external-link.sunan.me/blog/210522/L5aFKe3Hi7.png?imageslim)
![mark](https://external-link.sunan.me/blog/210522/j127cGFIhB.png?imageslim)

验证集正确率为0.97

## 杂项

### 验证
```Python
    def dev(self):
        acc_arr = []
        for i in range(len(self.Test) // 128):
            X, Y = self.getBatchTrain(is_train=False, batch_size=128, offset=i * 128)
            y_true = np.argmax(Y, axis=0)
            test_acc = self.getAcc(X, y_true)
            acc_arr.append(test_acc)
        print("验证集准确率为：" + str(np.mean(acc_arr)))
```

### 运行
```Python
if __name__ == '__main__':
    epoch = 15
    lr = 0.01
    batch_size = 128

    mnist = Mnist("data/mnist_all.mat")
    mnist.initParam()
    mnist.train(epoch, batch_size, lr)
    mnist.saveModel("cnnModel3.para")
    # mnist.loadModel("cnnModel2.para")
    mnist.dev()

```

## 项目代码已上传Github
> https://github.com/mofengboy/fcn-base-numpy
