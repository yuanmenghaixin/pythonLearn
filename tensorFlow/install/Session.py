
#会话的作用是处理内存分配和优化，使我们能够实际执行由计算图指定的计算。你可以将计算图想象为我们想要执行的计算的「模版」：它列出了所有步骤。为了使用计算图，我们需要启动一个会话，它使我们能够实际地完成任务；例如，遍历模版的所有节点来分配一堆用于存储计算输出的存储器。为了使用 TensorFlow 进行各种计算，你既需要计算图也需要会话。
#会话包含一个指向全局图的指针，该指针通过指向所有节点的指针不断更新。这意味着在创建节点之前还是之后创建会话都无所谓。
#创建会话对象后，可以使用 sess.run(node) 返回节点的值，并且 TensorFlow 将执行确定该值所需的所有计算。
import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print(sess.run(sum_node))


import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print(sess.run([two_node, sum_node]))