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

x2 = tf.constant(1, dtype=tf.float32)
y2 = tf.constant(-1, dtype=tf.float32)

#Equation of a line is y = mx + c
#y1 = mx1 + c, y2 = mx2 + c -- OR -- y1 - mx1 = c, y2 - mx2 = c;
#(1/c)y1 - (m/c)x1 = 1,(1/c)y2 - (m/c)x2 = 1;
#[(-m/c), (1/c)] * [x1, x2; y1, y2] = [1, 1] == AX = B ~ A = BX^-1

points     = tf.stack([[x1, x2], [y1, y2]])
B          = tf.ones([1, 2], dtype=tf.float32)
A_elements = tf.matmul(B, tf.matrix_inverse(points))

#Using "with ... as..." closes the Session automatically once it 
#is outside the loop. 
with tf.Session() as sess:
    A = sess.run(A_elements)
    c = 1/A[0][1]
    m = -c*A[0][0]
    print("Equation: y = {m}x + {c}".format(m=m, c=c))