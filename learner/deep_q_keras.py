import sys
sys.path.append('game/')
sys.path.append('learner/')

import os
import cv2
import math  
import numpy as np
import random as rand
import tensorflow as tf
import tensorflow.keras as keras
import game.flappy as flappy
import matplotlib.pyplot as plt
from collections import deque

'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    deep_q.py
Author:  Whitney King
Date:    March 9, 2019

References:

    Deep Learning Video Games 
    Author: Akshay Srivatsan
    github.com/asrivat1/DeepLearningVideoGames/blob/master/\
        deep_q_network.py

    Using Deep-Q Network to Learn to Play Flappy Bird
    Author: Kevin Chen
    github.com/yenchenlin/DeepLearningFlappyBird

    Visualizing Neural Network Activation
    Author: Arthur Juliani
    medium.com/@awjuliani/visualizing-neural-network-layer-activation-\
        tensorflow-tutorial-d45f8bf7bbc4

    Udacity ML Engineer Quadcopter Project
    github.com/WhitneyOnTheWeb/deep-learning-master/blob/master/\
        Quadcopter/tasks/takeoff.py
'''

class DeepQ:
    def __init__(self, lr = 0.01,
                 state_size = 4,
                 action_size = 2, 
                 hidden_size = 64,
                 name = 'DeepQ'):
        self.model = keras.Sequential()
        self.name  = name
        self.lr    = lr
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
        [filter.width, filter.height, channels.input, channels.output] '''

        '''---Create state from inputs---'''
        self.state   = tf.placeholder(tf.float32, 
                                      [None, 80, 80, self.S], 
                                       name = 'state')

        '''---Convolutional Neural Network Architecture---

        * Play arounds with these layers to see what works best
        * Don't forget to change references when creating layers
            * Each transformation should reference the previous one
            
        !!! tf.contrib will be removed in TensorFlow 2.0 !!!
            * Using keras.layers APIs to construct architecture instead
        '''
        #---Define MaxPooling2D Structure---
        MaxPooling2D = keras.layers.MaxPooling2D(pool_size = [1, 2, 2, 1],
                                                 strides = [1, 2, 2, 1],
                                                 padding = 'same')

        '''---Sequence 1: Convolution2D w/ ReLu Activation & MaxPooling2D---'''
        self.conv1 = keras.Sequential([
            # [80 x 80 x 4] -> [8, 8, 4, 64] -> [20 x 20 x 64] 
            keras.layers.Conv2D(filters = self.H,
                                kernel_size = [8, 8, self.S, self.H],
                                strides = 4,
                                use_bias = True,
                                bias_initializer = [self.H],
                                padding = 'same'),
            keras.layers.ReLU(),
            # [20 x 20 x 64] -> [1, 2, 2, 1] -> [10 x 10 x 64]
            MaxPooling2D
        ])

        '''---Sequence 2: Convolution2D w/ ReLu Activation & MaxPooling2D---'''  
        self.conv2 = keras.Sequential([
            # [10 x 10 x 64] -> [4, 4, 64, 128] -> [5 x 5 x 128]    
            keras.layers.Conv2D(filters = self.H * 2,
                                kernel_size = [4, 4, self.H, self.H * 2],
                                strides = 2,
                                use_bias = True,
                                bias_initializer = [self.H * 2],
                                padding = 'same'),
            keras.layers.ReLU(),
            # [5 x 5 x 128] -> [1, 2, 2, 1] -> [3 x 3 x 128]    
            MaxPooling2D
        ])

        '''---Sequence 3: Convolution2D w/ ReLu Activation & MaxPooling2D---'''
        self.conv3 = keras.Sequential([
            # [3 x 3 x 128] -> [3, 3, 128, 64] -> [3 x 3 x 64]    
            keras.layers.Conv2D(filters = self.H,
                                kernel_size = [3, 3, self.H * 2, self.H],
                                strides = 1,
                                use_bias = True,
                                bias_initializer = [self.H],
                                padding = 'same'),
            keras.layers.ReLU(),
            # [3 x 3 x 64] -> [1, 2, 2, 1] -> [2 x 2 x 64]
            MaxPooling2D
        ])      

        '''---Sequence 4: Flatten and Fully Connect ReLU Layers for Output---'''
        # [-1] adjusts size as needed for flattening
        self.out = keras.Sequential([
            # [2 x 2 x 64] -> [1, 256]
            keras.layers.Flatten(),
            keras.layers.ReLU(),
            # [1, 256] -> [None, 2]
            keras.layers.Dense(self.A, 
                               activation = 'relu')
        ])

        #---Compile all layers as single model---
        self.model.add(self.conv1)
        self.model.add(self.conv2)
        self.model.add(self.conv3)
        self.model.add(self.out)

        def visualize_layer(model)

    def forward(self, x):
        '''pass input into model, return results'''
        fc = self.model(x)
        out = self.out(fc.view(fc.size(0), -1))
        return x, out, fc