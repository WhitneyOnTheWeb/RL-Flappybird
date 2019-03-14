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
sys.path.append('/game/')

import os
import cv2
import tensorflow as tf
import random as rand  
import numpy as np
import game.flappy as flappy
from collections import deque

class DeepQ:
    def __init__(self, lr = 0.01,
                 state_size = 4,
                 action_size = 2, 
                 hidden_size = 64,
                 name = 'DeepQ'):
        self.name = name
        self.lr   = lr
        self.S = state_size
        self.A = action_size
        self.H = hidden_size

        '''---Deep-Q Neural Network Architecture---

        * Defines input layer
        * Set weight/bias of layer convolutions

        Convolution Formula for Outputs:
        * layer.width and layer.height are the same when the input is square

        layer.width = 
            ((input.width - filter.width + 2 * padding) / stride) + 1

        layer.height = 
           ((input.height - filter.height + 2 * padding) / stride) + 1

        ---Convolution Kernel Template---
        [filter.width, filter.height, channels, filter.number] '''

        '''---Create and reshape input layer---'''
        self.inputs   = tf.placeholder(tf.float32, 
                                      [None, self.S], 
                                       name = 'inputs')

        '''---Convolution 1---'''                              # stride = 4
        self.conv1_w = self.w_var([8, 8, 4, self.H])           # kernel
        self.conv1_b = self.b_var([self.H])

        '''---Convolution 2---'''                              # stride = 2
        self.conv2_w = self.w_var([4, 4, self.H, self.H * 2])  # kernel
        self.conv2_b = self.b_var([self.H * 2])

        '''---Convolution 3---'''                              # stride = 1
        self.conv3_w = self.w_var([3, 3, self.H * 2, self.H])  # kernel
        self.conv3_b = self.b_var([self.H])

        '''---Flatten 1---'''                                  # 256 x 1
        self.fc1_w   = self.w_var([-1, 256])                   # kernel
        self.fc1_b   = self.b_var([256])

        '''---Flatten 2---'''                                
        self.fc2_w   = self.w_var([256, self.A])               # kernel
        self.fc2_b   = self.b_var([self.A])                    # dimensions = 2

    def network(self):
            '''---Convolutional Neural Network Architecture---

            * Play arounds with these layers to see what works best
            * Don't forget to change the layer references when changing layers
              * Each layer should reference the previous one
            
            !!! tf.contrib will be removed in TensorFlow 2.0 !!!
            '''

            '''---Create Hidden Layers---'''
            '''---Convolution Layer 1: ReLu Activation ---'''
            #     [80 x 80 x 4] -> [8, 8, 4, 64] -> [20 x 20 x 64] 
            fc1 = tf.nn.relu(self.conv2d(self.inputs, 
                                         self.conv1_w, 4) + self.conv1_b)
            '''---Max Pool Output---'''
            #     [20 x 20 x 64] -> [1, 2, 2, 1] -> [10 x 10 x 64]
            pool1_h = self.max_pool(fc1)

            '''---Convolution Layer 2: ReLu Activation---''' 
            #     [10 x 10 x 64] -> [4, 4, 64, 128] -> [5 x 5 x 128]    
            fc2 = tf.nn.relu(self.conv2d(pool1_h, 
                                         self.conv2_w, 2) + self.conv2_b)
            '''---Max Pool Output---'''
            #     [5 x 5 x 128] -> [1, 2, 2, 1] -> [3 x 3 x 128]                           
            pool2_h = self.max_pool(fc2)

            '''---Layer 3: Convolution with ReLu Activation---'''
            #     [3 x 3 x 128] -> [3, 3, 128, 64] -> [3 x 3 x 64]    
            fc3 = tf.nn.relu(self.conv2d(pool2_h, 
                                         self.conv3_w, 1) + self.conv3_b)
            '''---Max Pool Output---'''
            #     [3 x 3 x 64] -> [1, 2, 2, 1] -> [2 x 2 x 64]
            pool3_h = self.max_pool(fc3)
            
            '''---Reshape and Flatten---                
                * [-1] instructs to adjust size as needed for flattening '''
            #     [2 x 2 x 64] -> [-1, 256] -> [256, 1]
            flat = tf.reshape(pool3_h, [-1, 256])    # flatten pool layer

            '''---Fully Connect ReLu Layers---'''
            fc1_h = tf.nn.relu(tf.matmul(flat, self.fc1_w) + self.fc1_b)

            '''---Create Linear Output Layer---
                * Dimension should be equal to the number of actions
                    - Flap
                    - Don't Flap '''
            #     [256, 1] -> [2, 1]
            out = tf.matmul(fc1_h, self.fc2_w) + self.fc2_b
            
            return self.inputs, out, fc1_h

    def w_var(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev = self.lr))

    def b_var(self, shape):
        return tf.Variable(tf.constant(self.lr, shape = shape))

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, 
                            strides = [1, stride, stride, 1], 
                            padding = "SAME")

    def max_pool(self, x):
        return tf.nn.max_pool(x, 
                            ksize = [1, 2, 2, 1], 
                            strides = [1, 2, 2, 1], 
                            padding = "SAME")

