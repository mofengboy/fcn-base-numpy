# 全连接网络关键点分析

## 核心矩阵
### 核心矩阵维度参考
1. W

W[l]的维度为（n[l],n[l-1]）,其中n代表为神经网络每一层的神经元。

2. B

B[l]的维度为（n[l]）。

### 图示
![mark](https://external-link.sunan.me/blog/210517/6DLiLfIGK7.png?imageslim)
如上图所示：

N[0] = 4

n[1] = 8

n[2] = 8

n[3] = 4

n[4] = 1

W[1]的维度为（n[1],n[0]）,即（8,4）

W[2]的维度为（n[2],n[1]）,即（8,8）

W[3]的维度为（n[3],n[2]）,即（4,8）

W[4]的维度为（n[4],n[3]）,即（1,4）

X的维度为（4,i）,其中4为每个样本的特征值数量，i为每次前后向传播时同时计算的样本的个数。

#### 前向传播

*以下计算忽略偏置b和激活函数,因为他们不对维度造成影响*

```
A = W×X+B

Z = activate(A)
```

这样计算后第一层Z[1]输出的维度为（8,4）×(4,i) = (8,i)

第二层Z[2]：（8,8）× (8,i) = (8,i)

第三层Z[3]：（4,8）× (8,i) = (4,i)

第四层Z[4]：（1,4）× (4,i) = (1,i)

#### 反向传播

损失层维度等于第四层激活层的维度等于第四层线性层的维度。

第四层：

dZ[4] = ∂Loss( Z4,y_ture) · ∂Z（A[4]），维度为（1，i）·(1,i) = (1,i)

dA[4] = dZ[4]· ∂A[4],维度为（1，i）· (1,i)=(1,i)

dW[4] = dA[4]×Z[3].T,维度为（1，i）×(i,4) = (1,4)

W[4] += dW[4],W[4]的维度为（1,4）

以下类似。

## 参数初始化
### 随机初始化
最简单的方法，但是也有弊端，一旦随机分布选择不当，就会陷入困境。

w = np.random.randn(m,n)

一般会乘以一个系数，保证参数初始不会太大。

### 高斯随机初始化
这种方法产生的不是均匀分布的随机数，均值通常选0，方差需要按经验人工选择。

w = np.random.normal(loc=0.0, scale=1.0, size=(m, n))

同样一般会乘以一个系数，保证参数初始不会太大。

### Xavier初始化
这种方法保证输入和输出的方差一致，这样就可以避免所有输出值都趋向于0，虽然刚开始的推导基于线性函数，但是在一些非线性神经元也很有效。

n_j为输入层的参数，n_(j+1)为输出层的参数.

公式如下：
![mark](https://external-link.sunan.me/blog/210518/5d5JA91LgL.png?imageslim)

np.random.randn(m, n) / np.sqrt(n / 2)

## 激活函数
1. sigmoid_forward

公式：$ sigmoid(x)=\frac{1}{1+e^{-x}} $

图像：![mark](https://external-link.sunan.me/blog/210518/GmlCgHCmHB.png?imageslim)

代码：

```Python
def sigmoid_forward(z):
    """
    sigmoid激活前向过程
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_backward(next_dz, z):
    """
    sigmoid激活反向过程
    :param next_dz:
    :param z:
    :return:
    """
    return sigmoid_forward(z) * (1 - sigmoid_forward(z)) * next_dz
```

2. tanh_forward

公式：$ tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=\frac{e^{2x}-1}{e^{2x}+1} $

图像：![mark](https://external-link.sunan.me/blog/210518/0DKa3e501K.png?imageslim)

代码：
```Python
def tanh_forward(z):
    """
    tanh激活前向过程
    :param z:
    :return:
    """
    return np.tanh(z)


def tanh_backward(next_dz, z):
    """
    tanh激活反向过程
    :param next_dz:
    :param z:
    :return:
    """
    return next_dz(1 - np.square(np.tanh(z)))
```

3. relu_forward

公式：$ y=\begin{cases} 
x,  & \mbox{if }x \ge0 \\
0, & \mbox{if }x < 0
\end{cases} $

图像：![mark](https://external-link.sunan.me/blog/210518/HcJjfbF92j.png?imageslim)

代码：

```Python
def relu_forward(z):
    """
    relu前向传播
    :param z: 待激活层
    :return: 激活后的结果
    """
    return np.maximum(0, z)


def relu_backward(next_dz, z):
    """
    relu反向传播
    :param next_dz: 激活后的梯度
    :param z: 激活前的值
    :return:
    """
    dz = np.where(np.greater(z, 0), next_dz, 0)
    return dz
```

4. elu_forward

公式：$ y=\begin{cases} 
x,  & \mbox{if }x \ge0 \\
a(e^x-1), & \mbox{if }x < 0
\end{cases}
\hspace {15 mm}
a>0$

图像：![mark](https://external-link.sunan.me/blog/210518/CCbj2imKa5.png?imageslim)

代码：

```Python
def elu_forward(z, alpha=0.1):
    """
    elu前向传播
    :param z: 输入
    :param alpha: 常量因子
    :return:
    """
    return np.where(np.greater(z, 0), z, alpha * (np.exp(z) - 1))


def elu_backward(next_dz, z, alpha=0.1):
    """
    elu反向传播
    :param next_dz: 输出层梯度
    :param z: 输入
    :param alpha: 常量因子
    :return:
    """
    return np.where(np.greater(z, 0), next_dz, alpha * next_dz * np.exp(z))
```

## 损失函数

### 均方误差损失函数

公式：![mark](https://external-link.sunan.me/blog/210518/C89BgifIc6.png?imageslim)


均方误差损失分数可能出现梯度下降缓慢，甚至梯度消失等问题。

### 交叉熵损失函数

公式：![mark](https://external-link.sunan.me/blog/210518/djeflh0330.png?imageslim)

其中：
- M表示为类别的数量
- Y指示变量（0或1）,如果该类别和样本i的类别相同就是1，否则是0
- p对于观测样本i属于类别c的预测概率

### 代码如下

由于求玩损失函数后，会紧接着求梯度，所以这里写在了一起。


```python
def mean_squared_loss(y_predict, y_true):
    """
    均方误差损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值
    :return:
    """
    loss = np.mean(np.sum(np.square(y_predict - y_true), axis=-1))  # 损失函数值
    dy = y_predict - y_true  # 损失函数关于网络输出的梯度
    return loss, dy


def cross_entropy_loss(y_predict, y_true):
    """
    交叉熵损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值,shape(N,d)
    :return:
    """
    y_predict = y_predict.T
    y_true = y_true.T
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    dy = y_probability - y_true
    return loss, dy.T
```

# [神经网络底层原理]-使用基于numpy的全连接神经网络识别mnist数据集
## 数据读取部分
### 数据集介绍
> 数据集地址： https://cs.nyu.edu/~roweis/data.html

数据包括从 "0 "到 "9 "的8位灰度图像；每类约6K训练实例；1K测试实例。
### 数据读取函数

ps:由于训练集是按照0-9,每类6K顺序排列，所以我这里将数据读入numpy数组后，对训练集的顺序进行了打乱操作（shuffle）,这是因为固定的数据集顺序，严重限制了梯度优化方向的可选择性，导致收敛点选择空间严重变少，容易导致过拟合。


```python
# 读入minst训练集数据
    def loadData(self, file_path):
        # 使用scipy读入mat文件数据
        mnist_all = sio.loadmat(file_path)
        train_raw = []
        test_raw = []
        # 依次读入数据集0-9
        for i in range(10):
            train_temp = mnist_all["train" + str(i)]
            for j in train_temp:
                train_raw.append([j, i])
        for i in range(10):
            test_temp = mnist_all["test" + str(i)]
            for j in test_temp:
                test_raw.append([j, i])

        # 随机打乱数据
        Train = np.array(train_raw)
        np.random.shuffle(Train)
        Test = np.array(test_raw)
        np.random.shuffle(Test)
        self.Train = Train
        self.Test = Test
        
        # 批量获取训练数据集
    def getBatchTrain(self, batch_size=128, offset=0):
        data = self.Train
        X = []
        Y = []
        y_temp = np.eye(10)
        length = data.shape[0]
        for i in range(batch_size):
            X.append(data[(offset + i) % length][0])
            Y.append(y_temp[data[(offset + i) % length][1]])
        X = np.array(X).T
        Y = np.array(Y).T
        #     归一化
        X = X / 225.0
        return X, Y
```

## 参数初始化部分
这里我采用的是xavier初始化方法。


```python
def initParam(self, init_type="xavier"):
    # 初始化W
    if init_type == "xavier":
        self.Parameters["W1"] = np.random.randn(256, 784) / np.sqrt(256 / 2)
        self.Parameters["W2"] = np.random.randn(128, 256) / np.sqrt(128 / 2)
        self.Parameters["W3"] = np.random.randn(10, 128) / np.sqrt(10 / 2)
    elif init_type == "normal":
        self.Parameters["W1"] = np.random.normal(loc=0.0, scale=1.0, size=(256, 784)) * 0.01
        self.Parameters["W2"] = np.random.normal(loc=0.0, scale=1.0, size=(128, 256)) * 0.01
        self.Parameters["W3"] = np.random.normal(loc=0.0, scale=1.0, size=(10, 128)) * 0.01
    elif init_type == "rand":
        self.Parameters["W1"] = np.random.rand(256, 784) * 0.01
        self.Parameters["W2"] = np.random.rand(128, 256) * 0.01
        self.Parameters["W3"] = np.random.rand(10, 128) * 0.01
    else:
        raise Exception("无效的参数初始化类型")

    # 初始化b
    self.Parameters["B1"] = np.zeros((256, 1))
    self.Parameters["B2"] = np.zeros((128, 1))
    self.Parameters["B3"] = np.zeros((10, 1))
```

## 前向传播部分

### 线性层
```Python
def fc_forward(w, x, b):
    return np.dot(w, x) + b
```

### 激活层
本次是使用的relu激活函数，代码见上述。

### 前向传播部分代码


```python
    def forward(self, train_data):
        self.nodes["A1"] = layer.fc_forward(self.Parameters["W1"], train_data, self.Parameters["B1"])
        self.nodes["Z1"] = activations.relu_forward(self.nodes["A1"])
        self.nodes["A2"] = layer.fc_forward(self.Parameters["W2"], self.nodes["Z1"], self.Parameters["B2"])
        self.nodes["Z2"] = activations.relu_forward(self.nodes["A2"])
        self.nodes["A3"] = layer.fc_forward(self.Parameters["W3"], self.nodes["Z2"], self.Parameters["B3"])
        self.nodes["Z3"] = activations.relu_forward(self.nodes["A3"])
        self.nodes["y"] = np.argmax(self.nodes["A3"], axis=0)
        return self.nodes["y"]
```

## 反向传播部分

### 损失函数
这里选择的是交叉熵损失函数。

### 求梯度

链式求导的流程如下图所示：
![mark](https://external-link.sunan.me/blog/210518/91mmJ5ka1k.png?imageslim)

```Python
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
```

### 代码


```python
    def backward(self, train_data, y_true):
        loss, self.gradients["A3"] = losses.cross_entropy_loss(self.nodes["A3"], y_true)
        self.gradients["W3"], self.gradients["B3"], self.gradients["Z2"] = \
            layer.fc_backward(self.gradients["A3"], self.Parameters["W3"], self.nodes["Z2"])

        self.gradients["A2"] = activations.relu_backward(self.gradients["Z2"].T, self.nodes["A2"])
        self.gradients["W2"], self.gradients["B2"], self.gradients["Z1"] = \
            layer.fc_backward(self.gradients["A2"], self.Parameters["W2"], self.nodes["Z1"])

        self.gradients["A1"] = activations.relu_backward(self.gradients["Z1"].T, self.nodes["A1"])
        self.gradients["W1"], self.gradients["B1"], self.gradients["Z1"] = \
            layer.fc_backward(self.gradients["A1"], self.Parameters["W1"], train_data)

        return loss
```

## 训练过程
### 代码
训练过程中每200次统计并输出损失和当前的准确率，最后输出验证集的正确率并画图。


```python
 def train(self, epoch, batch_size, lr):
        plt_x = []
        plt_acc = []
        plt_loss = []

        for i in range(epoch * 60000 // batch_size):
            X, Y = self.getBatchTrain(batch_size=batch_size, offset=i * batch_size)

            self.forward(X)
            loss = self.backward(X, Y)
            y_true = np.argmax(Y, axis=0)
            # print("y_true:")
            # print(y_true)

            # 参数更新
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

### 结果
1. 参数：

epoch = 5
    
lr = 0.01
    
batch_size = 128

结果：
![mark](https://external-link.sunan.me/blog/210518/m4KB47aH55.png?imageslim)
![mark](https://external-link.sunan.me/blog/210518/3H6A6B9kdF.png?imageslim)

验证集正确率为：0.9464

2. 参数：

epoch = 15
    
lr = 0.01
    
batch_size = 128

结果：
![mark](https://external-link.sunan.me/blog/210518/7balc6cJ03.png?imageslim)
![mark](https://external-link.sunan.me/blog/210518/Egd0L11bCb.png?imageslim)

验证集正确率为：0.9745
    

## 杂项
### 保存模型参数

```Python
    def saveModel(self, file_path):
        pickle.dump(self.Parameters, open(file_path, 'wb'))
```

### 读取模型参数

```Python
    def loadModel(self, file_path):
        self.Parameters = pickle.load(open(file_path, 'rb'))
```

### 获取准确率

```Python
    def getAcc(self, test_data, y_true):
        y_predict = self.forward(test_data)
        acc = np.mean(y_predict == y_true)
        return acc
```

### 验证

```Python
    def dev(self):
        acc_arr = []
        for i in range(len(self.Test) // 10000):
            X, Y = self.getBatchTrain(batch_size=10000, offset=i * 10000)
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
    # mnist.saveModel("model3.para")
    # mnist.loadModel("model1.para")
    mnist.dev()
```

## 项目代码已上传Github
> https://github.com/mofengboy/fcn-base-numpy
