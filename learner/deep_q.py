import math
import time
import numpy as np
import random as rand  
import tensorflow as tf
import keras_metrics as km
import tensorflow.keras as keras
from keras.optimizers import SGD, Adam
from keras.models import model_from_json, Sequential, Model
from keras.layers import Conv2D, MaxPool2D, ReLU, Dense, Flatten, InputLayer

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
    def __init__(self, 
                 S = 4,
                 A = 2, 
                 H = 64,
                 lr = 0.001):

        super(DeepQ, self).__init__()  
        self.nn = self._assemble(S, A, H, lr)

    def _assemble(self, S, A, H, lr):
        '''---Deep-Q Neural Network Architecture---

        * Defines input layer
        * Set weight/bias of layer convolutions

        Convolution Formula for Outputs:
        * layer.width and layer.height are the same when the input is square

        layer.width = 
            ((input.width - filter.width + 2 * padding) / stride) + 1

        ---Convolution Kernel Template---
        [filter.width, filter.height, channels.input, channels.output] '''
        
        '''---Convolution Layer and Sequence Architecture---

        * Play arounds with these layers to see what works best
        * Don't forget to change references when creating layers
            * Each transformation should reference the previous one

        '''
        shape = (80, 80, S)
        model = Sequential()
        MaxPooling2D = MaxPool2D(pool_size = (2, 2),
                                 strides = (2, 2),
                                 padding = 'same')
            # [?, 80, 80, 4] -> [20 x 20 x 64]
        model.add(InputLayer(input_shape = shape))
        model.add(Conv2D(filters = H,
                  kernel_size = (8, 8),
                  strides = (4, 4),
                  use_bias = True,
                  bias_initializer = keras.initializers.Constant(value = H),
                  padding = 'same',
                  kernel_regularizer = keras.regularizers.l2(0.0001)))
        model.add(ReLU())
            # [20 x 20 x 64] -> [1, 2, 2, 1] -> [10 x 10 x 64]
        model.add(MaxPooling2D)
            # [10 x 10 x 64] -> [4, 4, 64, 128] -> [5 x 5 x 128]    
        model.add(Conv2D(filters = H * 2,
                  kernel_size = (4, 4),
                  strides = (2, 2),
                  use_bias = True,
                  bias_initializer = keras.initializers.Constant(value = H * 2),
                  padding = 'same',
                  kernel_regularizer = keras.regularizers.l2(0.001)))
        model.add(ReLU())
            # [5 x 5 x 128] -> [1, 2, 2, 1] -> [3 x 3 x 128]    
        model.add(MaxPooling2D)

        model.add(Conv2D(filters = H,
                  kernel_size = (3, 3),
                  strides = (1, 1),
                  use_bias = True,
                  bias_initializer = keras.initializers.Constant(value = H),
                  padding = 'same',
                  kernel_regularizer = keras.regularizers.l2(0.01)))
        model.add(ReLU())
            # [3 x 3 x 64] -> [1, 2, 2, 1] -> [2 x 2 x 64]
        model.add(MaxPooling2D)
            # [2 x 2 x 64] -> [1, 256]
        model.add(Flatten())
        model.add(ReLU())

        '''---Sequence 4: Flatten and Fully Connect ReLU Layers for Output---
            * Dimension should be equal to the number of actions
                - Flap
                - Don't Flap'''
            # [1, 256] -> [None, 2]
        model.add(Dense(A, activation = 'linear'))

        '''---Compile the model for use---'''
        acc  = 'accuracy'
        prec = km.binary_precision()
        re   = km.binary_recall()
        f1   = km.binary_f1_score()
        tp   = km.binary_true_positive()
        tn   = km.binary_true_negative()
        fp   = km.binary_false_positive()
        fn   = km.binary_false_negative()

        model.compile(loss = 'mse', 
                      optimizer = Adam(lr = lr), 
                      metrics = [acc, prec, re, f1, tp, tn, fp, fn])
        return model