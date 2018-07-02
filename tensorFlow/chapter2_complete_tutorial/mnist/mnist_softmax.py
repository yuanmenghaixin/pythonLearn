"""A very simple MNIST classifer（分类器）.

See extensive documentation at ??????? (insert public URL)
"""
from __future__ import print_function

# Import data
from tensorFlow.chapter2_complete_tutorial.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

# Create the model 创建模型
x = tf.placeholder("float", [None, 784])
### x（图片的特征值）：这里使用了一个28*28=784列的数据来表示一个图片的构成，也就是说，每一个点都是这个图片的一个特征，
# 这个其实比较好理解，因为每一个点都会对图片的样子和表达的含义有影响，只是影响的大小不同而已。
# 至于为什么要将28*28的矩阵摊平成为一个1行784列的一维数组，我猜测可能是因为这样做会更加简单直观。
W = tf.Variable(tf.zeros([784, 10]))  # w表示每一个特征值（像素点）会影响结果的权重
##### W（特征值对应的权重）：这个值很重要，因为我们深度学习的过程，就是发现特征，经过一系列训练，从而得出每一个特征对结果影响的权重，
# 我们训练，就是为了得到这个最佳权重值。
b = tf.Variable(tf.zeros([10]))
### b（偏置量）：是为了去线性话（我不是太清楚为什么需要这个值）
# y = tf.nn.softmax(tf.matmul(x,W) + b)  # this will be lead an error because of log(0)
y = tf.nn.log_softmax(tf.matmul(x, W) + b)
#### y（预测的结果）：单个样本被预测出来是哪个数字的概率，
# 比如：有可能结果是[ 1.07476616 -4.54194021 2.98073649 -7.42985344 3.29253793 1.96750617 8.59438515 -6.65950203 1.68721473 -0.9658531 ]，
# 则分别表示是0，1，2，3，4，5，6，7，8，9的概率，然后会取一个最大值来作为本次预测的结果，对于这个数组来说，结果是6（8.59438515）

# Define loss and optimizer
y_ = tf.placeholder("float", [None, 10])
### y_（真实结果）：来自MNIST的训练集，每一个图片所对应的真实值，如果是6，则表示为：[0 0 0 0 0 1 0 0 0]
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = -tf.reduce_sum(y_ * y)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
for i in range(1000):
    ###再下面两行代码是损失函数（交叉熵）和梯度下降算法，通过不断的调整权重和偏置量的值，
    # 来逐步减小根据计算的预测结果和提供的真实结果之间的差异，以达到训练模型的目的。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
