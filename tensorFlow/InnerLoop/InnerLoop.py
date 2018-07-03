# 在深度学习中，典型的「内循环」训练如下：
#
# 1. 获取输入和 true_output
# 2. 根据输入和参数计算「推测」值
# 3. 根据推测与 true_output 之间的差异计算「损失」
# 4. 根据损失的梯度更新参数
#
# 让我们把所有东西放在一个快速脚本里，解决简单的线性回归问题：
import random

import tensorflow as tf
### build the graph## first set up the parameters
m = tf.get_variable("m", [], initializer=tf.constant_initializer(0.))
b = tf.get_variable("b", [], initializer=tf.constant_initializer(0.))
init = tf.global_variables_initializer()
## then set up the computations
input_placeholder = tf.placeholder(tf.float32)
output_placeholder = tf.placeholder(tf.float32)

x = input_placeholder
y = output_placeholder
y_guess = m * x + b

loss = tf.square(y - y_guess)
## finally, set up the optimizer and minimization node
#class tf.train.Optimizer基本的优化类，该类不常常被直接调用，而较多使用其子类，
# 比如GradientDescentOptimizer, AdagradOptimizer
# 或者MomentumOptimizer
print(1e-3)
optimizer = tf.train.GradientDescentOptimizer(1e-3)#1e-3=0.001不会向计算图中添加节点。它只是创建一个包含有用的帮助函数的 Python 对象
train_op = optimizer.minimize(loss)#将一个节点添加到图中，并将一个指针存储在变量 train_op 中。train_op 节点没有输出，但是有一个十分复杂的副作用：
### start the session
sess = tf.Session()
sess.run(init)
### perform the training loop*import* random
## set up problem
true_m = random.random()
true_b = random.random()
for update_i in range(100):
 ## (1) get the input and output
 input_data = random.random()
 output_data = true_m * input_data + true_b

 ## (2), (3), and (4) all take place within a single call to sess.run()!
 _loss, _ = sess.run([loss, train_op], feed_dict={input_placeholder: input_data, output_placeholder: output_data})
 print(update_i,input_data,output_data,true_m,true_b, _loss)
### finally, print out the values we learned for our two variables*print* "True parameters: m=%.4f, b=%.4f" % (true_m, true_b)*print* "Learned parameters: m=%.4f, b=%.4f" % tuple(sess.run([m, b]))