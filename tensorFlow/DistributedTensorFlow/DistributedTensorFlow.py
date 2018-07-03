# Start a TensorFlow server as a single-process "cluster".
#tf.train.Server.create_local_server() 会在本地创建一个单进程集群，该集群中的服务默认为启动状态。
# 其中server.target的格式为：’grpc://localhost:port’，port是一个整数，为随机分配的端口。
import tensorflow as tf
c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)  # Create a session on the server.
print(sess.run(c))

import tensorflow as tf
c = tf.constant("Hello, distributed TensorFlow!")
sess = tf.Session("grpc://localhost:2222")
print(sess.run(c))