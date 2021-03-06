
#什么是计算图？它本质上是一个全局数据结构,导入 TensorFlow 并不会给我们生成一个有趣的计算图。而只是一个单独的，空白的全局变量。
import tensorflow as tf
#快看！我们得到了一个节点。它包含常量 2。很惊讶吧，这来自于一个名为 tf.constant 的函数。当我们打印这个变量时，我们看到它返回一个 tf.Tensor 对象，它是一个指向我们刚刚创建的节点的指针。为了强调这一点，以下是另外一个示例：
two_node = tf.constant(2)
#每次我们调用 tf.constant 时，我们都会在图中创建一个新的节点。即使该节点的功能与现有节点相同，即使我们将节点重新分配给同一个变量，或者即使我们根本没有将其分配给一个变量，结果都是一样的。
print(two_node)