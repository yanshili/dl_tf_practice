#!usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001  # learning rate
training_iters = 10000
batch_size = 128

n_inputs = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

# define biases
biases = {
    # (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for inputs to cell
    #####################################################################
    # X(128 batch, 28 steps, 28 inputs)
    # ==>(128*28, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in ==> (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 28 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    #####################################################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm is divided into two parts (c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state,
                                        time_major=False)  # 输入序列是否放在了主要维度

    # hidden layer for outputs as the final results
    #####################################################################
    results = tf.matmul(states[1], weights['out']) + biases['out']

    # or

    # unpack to list[(batch, outputs)...] * steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


prediction = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train_step = tf.train.AdamOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
        session.run([train_step], feed_dict={
            x: batch_x,
            y: batch_y,
        })
        if step % 20 == 0:
            print(session.run(accuracy, feed_dict={
                x: batch_x,
                y: batch_y,
            }))
