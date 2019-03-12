'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    deep_q.py
Author:  Whitney King
Date:    March 9, 2019

References:

    Deep Learning Video Games 
    Author: Akshay Srivatsan
    github.com/asrivat1/DeepLearningVideoGames/blob/master/deep_q_network.py

    Using Deep-Q Network to Learn to Play Flappy Bird
    Author: Kevin Chen
    github.com/yenchenlin/DeepLearningFlappyBird

    Udacity ML Engineer Quadcopter Project
    github.com/WhitneyOnTheWeb/deep-learning-master/blob/master/Quadcopter/tasks/takeoff.py
'''
import sys
sys.path.append('../game/')

import os
import cv2
import opencv
import tensorflow as tf
import random as rand  
import numpy as np
import flappy
from collections import deque

class DeepQ:
    def __init__(self, lr = 0.01,
                 state_size = 4,
                 action_size = 2, 
                 hidden_size = 64,
                 name = 'DeepQ'):
        self.name = name
        self.lr   = lr
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

    def network(self):
        with tf.variable_scope(self.name):
            # Deep-Q Neural Network Architecture-------------------------------
            # Preprocessing----------------------------------------------------
            # Set weight/bias of transformation on each network layer
            conv1_w = w_var([8, 8, 4, 32])
            conv1_b = b_var([32])
            conv2_w = w_var([4, 4, 32, 64])
            conv2_b = b_var([64])
            conv3_w = w_var([3, 3, 64, 64])
            conv3_b = b_var([64])
            fc1_w   = w_var([1600, 512])
            fc1_b   = b_var([512])
            fc2_w   = w_var([512, self.action_size])
            fc2_b   = b_var([self.action_size])

            # Create and reshape input layer
            inputs    = tf.placeholder(tf.float32, 
                                       [None, 80, 80, 4], 
                                       name = 'inputs')

            # Encode actions for Q-Value comparisons
            actions_  = tf.placeholder(tf.int32, 
                                       [None], 
                                       name = 'actions')
            actions   = tf.one_hot(actions_, self.action_size)

            '''
            DeepQ Network Layer Architecture

            * Play arounds with these layers to see what works best
            * Don't forget to change the layer references when chancing layers
              * Each layer should reference the previous
            '''
            # Create ReLu Hidden Layers----------------------------------------
            fc1 = tf.contrib.layers.fully_connected(inputs, self.hidden_size)   # this one? Add activation function?
            #conv1_h = tf.nn.relu(conv2d(inputs, conv1_w, 4) + conv1_b)         # Or this one?
            pool1_h = max_pool_2x2(fc1)                                     

            fc2 = tf.contrib.layers.fully_connected(pool1_h, self.hidden_size)
            #conv2_h = tf.nn.relu(conv2d(pool1_h, conv2_w, 2) + conv2_b)

            #pool2_h = max_pool_2x2(conv2_h)
            conv3_h = tf.nn.relu(conv2d(fc2, conv3_w, 1) + conv3_b)
            #pool3_h = max_pool_2x2(conv3_h)
            #pool3_flat = tf.reshape(pool3_h, [-1, 256])
            conv3_flat = tf.reshape(conv3_h, [-1, 1600])

            fc1_h = tf.nn.relu(tf.matmul(conv3_flat, fc1_w) + fc1_b)

            # Create linear output layer
            out = tf.matmul(fc1_h, fc2_w) + fc2_b
            #out = tf.contrib.layers.fully_connected(fc2,                
            #                            self.action_size,
            #                            activation_fn = None)
            #------------------------------------------------------------------
            return inputs, out, fc1_h

    def w_var(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

    def b_var(self, shape):
        return tf.Variable(tf.constant(0.01, shape = shape))

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, 
                            strides = [1, stride, stride, 1], 
                            padding = "SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, 
                            ksize = [1, 2, 2, 1], 
                            strides = [1, 2, 2, 1], 
                            padding = "SAME")

