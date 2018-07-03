# 一个更有价值的应用可能涉及构建一个计算图，它接受输入，以某种（一致）方式处理它，并返回一个输出。
# 最直接的方法是使用占位符。占位符是一种用于接受外部输入的节点。
import tensorflow as tf

input_placeholder = tf.placeholder(tf.int32)
three_node = tf.constant(3)
sum_node = input_placeholder + three_node
sess = tf.Session()
print(sess.run(three_node,feed_dict={input_placeholder: 2}))
try:
    print(sess.run(sum_node,feed_dict={input_placeholder: 'p'}))
except ValueError:
    print('NameError')