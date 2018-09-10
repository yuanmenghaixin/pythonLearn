# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#load_mnist(normalize=True,flatten=True, one_hot_label=False) 这样，设置3 个参数。第1 个参数
# normalize设置是否将输入图像正规化为0.0～1.0 的值。如果将该参数设置
# 为False，则输入图像的像素会保持原来的0～255。第2 个参数flatten设置
# 是否展开输入图像（变成一维数组）。如果将该参数设置为False，则输入图
# 像为1 × 28 × 28 的三维数组；若设置为True，则输入图像会保存为由784 个
# 元素构成的一维数组。第3 个参数one_hot_label设置是否将标签保存为onehot
# 表示（one-hot representation）。one-hot 表示是仅正确解标签为1，其余
# 皆为0 的数组，就像[0,0,1,0,0,0,0,0,0,0]这样。当one_hot_label为False时，
# 只是像7、2这样简单保存正确解标签；当one_hot_label为True时，标签则
# 保存为one-hot 表示。

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
# 输出各个数据的形状
#print(x_train.shape) # (60000, 784)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
