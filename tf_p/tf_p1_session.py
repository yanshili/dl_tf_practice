#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

# method1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# method2
# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)


input_x = [
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]

]

result = tf.transpose(input_x, perm=[0, 2, 1])  # 将第2维和第3维转换
"""
即(3, 4)矩阵
[1  2  3  4]
[5  6  7  8]
[9, 10, 11, 12]
转换为(4, 3)矩阵
[ 1  5  9]
[ 2  6 10]
[ 3  7 11]
[ 4  8 12]
"""

with tf.Session() as sess:
    print(input_x)
    print(sess.run(result))