# coding: utf-8
import sys, os
import numpy as np

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import sigmoid, softmax, cross_entropy_error, sigmoid_grad
from common.gradient import numerical_gradient


# 两层神经网络类
class TwoLayerNet:

    # 进行初始化。参数从头开始依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        # numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
        # numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # params['W1']是第1 层的权重
        self.params['b1'] = np.zeros(hidden_size)  # 返回全是0的数组#params['b1']是第1 层的偏置。
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # params['W2']是第2 层的权重
        self.params['b2'] = np.zeros(output_size)

    # 进行识别（推理）。参数x是图像数据
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 计算损失函数的值。参数x 是图像数据，t 是正确解标签（后面3 个方法的参数也一样）
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)  # 交叉熵

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 计算权重参数的梯度
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # 保存梯度的字典型变量（numerical_gradient()方法的返回值）。
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])  # grads['W1']是第1 层权重的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])  # grads['b1']是第1 层偏置的梯度。
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 计算权重参数的梯度。numerical_gradient()的高速版，将在下一章实现
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
