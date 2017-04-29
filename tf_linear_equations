#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:23:44 2017

@author: SkandaBharadwaj
"""
#%%Sloving linear equations 
#(To find the equation of a lin given 2 points)
import tensorflow as tf

#Define the two points (x1, y1) and (x2, y2)
x1 = tf.constant(2, dtype=tf.float32)
y1 = tf.constant(1, dtype=tf.float32)
point1 = tf.stack([x1, y1])

x2 = tf.constant(1, dtype=tf.float32)
y2 = tf.constant(-1, dtype=tf.float32)
point2 = tf.stack([x2, y2])

#In the formulation Ax = B find A by A = B(X^-1)
X = tf.transpose(tf.stack([point1, point2]))
b = tf.constant([[9, 3]], dtype=tf.float32)

parameters = tf.matmul(b, tf.matrix_inverse(X))

#Using "with ... as..." closes the Session automatically once it 
#is outside the loop. 
with tf.Session() as sess:
    A = sess.run(parameters)
    b = A[0][1]
    a = A[0][0]
    print("Equation: y = {a}x + {b}".format(a=a, b=b))