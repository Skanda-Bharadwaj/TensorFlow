#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:26:04 2017

@author: Skanda Bharadwaj

@topic: Basic MNIST classification using Softmax Regression
"""

#%% Download and fetch input data 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#%% Import libraries
import tensorflow as tf

#%% Create a placeholder and variables for input 
x = tf.placeholder(tf.float32, shape=[None, 784]) #here None means that a dimension can be of any length

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#%% Create Softmax Regression model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#%% Implement Cross-Entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) #Numerically more stable

#%% Minimize Cross-Entropy using an optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#%% Create Interactive Session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#%% Train the created model
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
#%% Evaluate model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%% Find efficiency on test data
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))