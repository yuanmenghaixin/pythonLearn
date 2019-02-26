"""Trains and Evaluates the MNIST network using a feed dictionary.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html

"""
from __future__ import print_function

# pylint: disable=missing-docstring
import time

import tensorflow as tf
from tensorFlow.chapter2_complete_tutorial.mnist import input_data
from tensorFlow.chapter2_complete_tutorial.mnist import mnist

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                        'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = int(data_set.num_examples / FLAGS.batch_size)
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / float(num_examples)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.
    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data) #，准备训练、验证和测试数据集。这里TensorFlow 提供了内置模块可以直接操作下载MNIST datasets 数据集。

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():#使用默认图（ graph), TensorFlow 里使用图来表示计算任务，图中的节点被称为Op (operation ），一个Op 获取0 个或多个tensor 执行计算，并产生0 个或多个tensor 。
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = mnist.inference(images_placeholder,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = mnist.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()#创建saver 来保存模型。

        # Create a session for running Ops on the Graph.
        sess = tf.Session() #创建会话（ session ）上下文，图需要在会话中运行。

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init) #，运行初始化所有变量，之前创建的Op 只是描述了数据是怎样流动或者怎么计算，没有真正开始执行运算，只有把Op 放入sess . run(Op）中才会开始运行。

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def) #创建SummaryWriter，把summary_op 返回的数据写到磁盘。

        # And then after everything is built, start the training loop.
        for step in range(FLAGS.max_steps): #开始训练循环总共运行FLAGS.max_steps 个step 。
            start_time = time.time() #，记录每个step 的开始时间

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)#取一个batch 训练数据，使用真实数据填充图片和标签占位符。

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)#把一个batch 数据放入模型进行训练，得到train_op （被忽略掉了）和loass op 的返回值，如果你想观察Op 或者变量的值，需要把它们放到列表里传给sess.run （），然后它们的值会以元组的形式返回。

            duration = time.time() - start_time#计算运行一个step 花费的时间

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:  #每100 个step 把summa可信息写入磁盘一次。
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)#每1000 个step 或者是最后一个step 保存一下模型，并且打印训练过程中产生的模型在训练、验证、测试数据集上的准确率。
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)


def main(_):
    run_training() #启动TensorFlow 后首先调用main 函数，判断目录是否存在，存在就删除不存在就创建。最后开始训练MNIST 数据。


if __name__ == '__main__': #解析命令行启动TensorFlow
    tf.app.run()     #解析命令行启动TensorFlow
