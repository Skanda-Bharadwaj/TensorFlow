# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:43:49 2017

@author: Skanda Bharadwaj

@topic: Basic units of TensorFlow
"""

#%% Import the libraries
import tensorflow as tf
import numpy as np

#%% tf.Variable example
y = tf.Variable([3], tf.int16)
model = tf.global_variables_initializer()
sess = tf.Session()
sess.run(model)
print(sess.run(y))

#%% tf.transpose example
xx = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(xx)
x = tf.Variable(xx, name='x')
model = tf.global_variables_initializer()
sess = tf.Session()
x = tf.transpose(x, perm=[1, 0])
sess.run(model)
result = sess.run(x)
print(result)

#%% tf.reverse_sequence example
xx = np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
height, width = xx.shape
print(xx)
x = tf.Variable(xx, name='x')
model = tf.global_variables_initializer()
sess = tf.Session()
x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
sess.run(model)
result = sess.run(x)
print(result)

#%% tf.placeholder example
x = tf.placeholder('float', [None, 3], name='x')
y = x*2
sess = tf.Session()
x_data = [[1, 2, 3], [4, 5, 6]];
print(sess.run(y, feed_dict={x: x_data}))

#%% tf.slice example
mat = np.array([[[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]]])

mat = tf.Variable(mat, name='mat')
model = tf.global_variables_initializer()
sess = tf.Session()
sess.run(model)
print(sess.run(mat), '\n')
print('result')
print(sess.run(tf.slice(mat, [0, 0, 0], [1, 1, 3])))
print(sess.run(tf.slice(mat, [1, 0, 0], [1, 2, 3])))
print(sess.run(tf.slice(mat, [1, 0, 0], [2, 2, 2])))

#%% tf.reshape example
x = np.array([[[1, 2, 1], [2, 3, 2]], 
              [[3, 4, 3], [4, 5, 4]], 
              [[5, 6, 5], [6, 7, 6]]])

x = tf.Variable(x, name='x')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.reshape(sess.run(x), [-1])))
print(sess.run(tf.reshape(sess.run(x), [-1, 2])))
print(sess.run(tf.reshape(sess.run(x), [2, -1])))