# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:05:12 2017

@author: Skanda Bharadwaj
"""

#%%Slicing an image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/Files/cat.jpg"

image = mpimg.imread(filename)
plt.imshow(image)
plt.show()

x = tf.placeholder("uint8", [None, None, 3])
sliced_image = tf.slice(x, [50, 400, 0], [1100, 800, -1])

sess = tf.Session()
result = sess.run(sliced_image, feed_dict={x:image})
print(result.shape)

plt.imshow(result)
plt.show()
