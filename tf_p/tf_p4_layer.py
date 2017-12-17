#!usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf


# add layer

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        out_puts = Wx_plus_b
    else:
        out_puts = activation_function(Wx_plus_b)
    return out_puts
