# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

X = np.arange(-20.0, 20.0, 0.5)
Y = softmax(X)
print(X)
print(Y)
plt.plot(X, Y)
plt.ylim(-0.1, 0.5)
plt.show()


