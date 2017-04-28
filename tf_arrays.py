# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:46:41 2017

@author: Skanda Bharadwaj
"""

#%% Arrays example
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/cat.jpg"

image = mpimg.imread(filename)
height, width, depth = image.shape
print("Image size =", image.shape)
print("Original image:")
plt.imshow(image)
plt.show()

#Creating a Tensor, model and a session
x = tf.Variable(image, name='x')
y = tf.Variable(image, name='x')
model = tf.global_variables_initializer()
sess = tf.Session()
x = tf.transpose(x, perm=[1, 0, 2])
y = tf.reverse_sequence(y, [width] * height, 1, batch_dim=0)
sess.run(model)
result1 = sess.run(x)
result2 = sess.run(y)

print("Transposed image:")  
plt.imshow(result1)
plt.show()

print("Reversed image:")  
plt.imshow(result2)
plt.show()