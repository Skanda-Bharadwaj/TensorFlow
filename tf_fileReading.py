# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:33:47 2017

@author: Skanda Bharadwaj
"""

#%% File operations
import tensorflow as tf
import numpy as np
import csv
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "\Files\pima.csv"

features = tf.placeholder(tf.float16, shape=[8], name='features')
country  = tf.placeholder(tf.float16, name='country')
total    = tf.reduce_sum(features, name='total')

printerop = tf.Print(total, [country, features, total], name='printer')


with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            f1, f2, f3, f4, f5, f6, f7, f8, f9 = line.strip().split(",")
            
            # Run the Print ob
            total = sess.run(printerop, feed_dict={features: [float(f2)], country:float(f1)})
            print(f1, f9)


















#%% For future reference
#with open(filename, 'r') as f:
#    reader = csv.reader(f)
#    data = []
#    for rows in reader:
#        row = []
#        for i in rows:
#            row.append(float(i))
#        data.append(row)
#f.close()
#
#data = np.array(data)
#x_data = data[:, 0:-1]
#y_data = data[:, -1]
#%%