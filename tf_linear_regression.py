# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:38:01 2017

@author: Skanda Bharadwaj
"""

#%% Linear Regression from tensorflow
import numpy as np
import tensorflow as tf

#Model Parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

#Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#Linear Model
linear_model = W*x + b

#Loss function 
loss = tf.reduce_sum(tf.square(linear_model-y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#Training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

#Training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})
    
#Evaluation of accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))



