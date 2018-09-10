# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1) #指定图中绘制的y轴的范围
plt.show()


#矩阵乘法
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(A.shape)
print(B.shape)
print("矩阵乘机dot：")
print(np.dot(A, B))

A = np.array([[1,2,3], [4,5,6]])
B = np.array([[1,2], [3,4], [5,6]])
print(A.shape)
print(B.shape)
print("矩阵乘机dot：")
print(np.dot(A, B))
##上个矩阵计算结果解析：
#A矩阵
# 1,2,3
# 4,5,6

#B矩阵
# 1,2
# 3,4
# 5,6
# A*B的结果
# 22,28
# 49,64
# A*B矩阵的由来
# 22=1*1+2*3+3*5
# 28=1*2+2*4+3*6
# 49=1*4+5*3+6*5
# 64=4*2+5*4+6*6=8+20+36
