# 创建变量，就需要使用 tf.get_variable()。tf.get_variable() 的前两个参数是必需的，其余参数是可选的
# 它们是 tf.get_variable(name，shape)。name 是一个唯一标识这个变量对象的字符串。它必须相对于全局图是唯一的，所以要明了你使用过的所有命名，确保没有重复。shape 是与张量形状对应的整数数组，它的语法非常直观：按顺序，每个维度只有一个整数。例如，一个 3x8 矩阵形状是 [3, 8]。要创建一个标量，就需要使用形状为 [] 的空列表。
import tensorflow as tf

# 另一个异常。当首次创建变量节点时，它的值基本上为「null」，并且任何试图对它求值的操作都会引发这个异常。我们只能在将值放入变量之后才能对其求值。主要有两种将值放入变量的方法：初始化器和 tf.assign()。我们先看看 tf.assign()：
count_variable = tf.get_variable("count", [])
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)  # 恒等运算。tf.assign(target, value) 不做任何有趣的运算，通常与 value 相等
sess = tf.Session()
sess.run(assign_node)
print(sess.run(count_variable))

# 问题出现在会话和图之间的分离。我们已将 get_variable 的 initializer 属性设置为指向 const_init_node，但它只是在图中的节点之间添加了一个新的连接。我们还没有做任何解决异常根源的事：与变量节点（存储在会话中，而不是计算图中）相关联的内存仍然设置为「null」。我们需要通过会话使 const_init_node 去更新变量。
import tensorflow as tf

const_init_node = tf.constant_initializer(0.)  # 初始化器
count_variable = tf.get_variable("count1", [], initializer=const_init_node)  # 初始化器
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run([count_variable]))

import tensorflow as tf
# 定义‘符号’变量，也称为占位符
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.multiply(a, b)  # 构造一个op节点
sess = tf.Session()  # 建立会话
# 运行会话，输入数据，并计算节点，同时打印结果
print(sess.run(y, feed_dict={a: 3, b: 3}))
# 任务完成, 关闭会话.
sess.close()

#用 tf.Print 调试
import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
answer, inspection = sess.run([sum_node, [two_node, three_node]])
print(inspection)
print(answer)

import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node### this new copy of two_node is not on the computation path, so nothing prints!
print_two_node = tf.Print(two_node, [two_node, three_node, sum_node])
sess = tf.Session()
print(sess.run(sum_node))