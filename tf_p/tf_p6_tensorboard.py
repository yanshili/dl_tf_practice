#!usr/bin/env python
# -*- coding: UTF-8 -*-

# 神经网络结构可视化

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

l1 = add_layer(xs, 1, 10, tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

session = tf.Session()

writer = tf.summary.FileWriter('tf_logs/', session.graph)

session.run(init)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()
#
# for i in range(1000):
#     session.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         print(session.run(loss, feed_dict={xs: x_data, ys: y_data}))
#         try:
#             ax.lines.remove(ax.lines[0])
#         except Exception:
#             pass
#         prediction_value = session.run(prediction, feed_dict={xs: x_data, ys: y_data})
#         lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
#         plt.pause(0.1)
