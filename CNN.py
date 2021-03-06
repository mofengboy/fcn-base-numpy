import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import layer
import losses
import activations
from PIL import Image


class Mnist:
    Train = []  # 训练集
    Test = []  # 验证集
    Parameters = {}  # 参数矩阵

    def __init__(self, file_path):
        self.loadData(file_path)
        self.nodes = {}
        self.gradients = {}

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

    def forward(self, train_data):
        # X1 = train_data[0]
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

    def getAcc(self, test_data, y_true):
        y_predict = self.forward(test_data)
        acc = np.mean(y_predict == y_true)
        return acc

    def saveModel(self, file_path):
        pickle.dump(self.Parameters, open(file_path, 'wb'))

    def loadModel(self, file_path):
        self.Parameters = pickle.load(open(file_path, 'rb'))

    def dev(self):
        acc_arr = []
        for i in range(len(self.Test) // 128):
            X, Y = self.getBatchTrain(is_train=False, batch_size=128, offset=i * 128)
            y_true = np.argmax(Y, axis=0)
            test_acc = self.getAcc(X, y_true)
            acc_arr.append(test_acc)
        print("验证集准确率为：" + str(np.mean(acc_arr)))

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
