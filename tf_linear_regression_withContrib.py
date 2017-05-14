# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:46:28 2017

@author: Skanda Bharadwaj

@topic: Linear Regression with contrib from tensorflow
"""
#%% Import libraries
import tensorflow as tf
import numpy as np

#%% Define features and estimators
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

#%% Define Inputs and model
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=2000)
#%% Estimate and evaluate model
estimator.fit(input_fn=input_fn, steps=2000)
estimator.evaluate(input_fn=input_fn)