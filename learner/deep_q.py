import math
import time
import numpy as np
import random as rand

from keras import metrics
import keras_metrics as km
from keras.optimizers import SGD, Adam, Adamax, Adadelta, RMSprop
from keras.regularizers import l2
from keras.initializers import Constant
from keras.models import model_from_json, Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, LeakyReLU
from keras.layers import Input, InputLayer, ReLU, Softmax, BatchNormalization
from keras.applications import VGG16, ResNet50
from keras.preprocessing import image

from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.agents.sarsa import SARSAAgent
from rl import memory, policy
from rl.util import huber_loss


'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    deep_q.py
Author:  Whitney King
Date:    March 9, 2019

References:

    Visualizing Neural Network Activation
    Author: Arthur Juliani
    medium.com/@awjuliani/visualizing-neural-network-layer-activation-\
        tensorflow-tutorial-d45f8bf7bbc4

    Udacity ML Engineer Quadcopter Project
    github.com/WhitneyOnTheWeb/deep-learning-master/blob/master/\
        Quadcopter/tasks/takeoff.py

    Keras Reinforcement Learning
    github.com/keras-rl/keras-rl
'''
# @misc{plappert2016kerasrl,
#       author = {Matthias Plappert},
#       title = {keras-rl},
#       year = {2016},
#       publisher = {GitHub},
#       journal = {GitHub repository},
#       howpublished = {\url{https://github.com/keras-rl/keras-rl}},
# }


class DeepQ:
    def __init__(self,
                 S=4,
                 A=2,
                 H=64,
                 lr=0.01,
                 loss='logcosh',
                 opt='adam',
                 model='custom'):

        super(DeepQ, self).__init__()
        self.shape = (80, 80, S)
        prec = km.binary_precision()
        re = km.binary_recall()
        f1 = km.binary_f1_score()
        #tp   = km.binary_true_positive()
        #tn   = km.binary_true_negative()
        #fp   = km.binary_false_positive()
        #fn   = km.binary_false_negative()

        if model == 'vgg16':
            self.nn = VGG16()
        if model == 'resnet50':
            self.nn = ResNet50()
        else:
            '''---Compile the model for use---'''
            self.nn = self._create_model(S, A, H, lr)
            if opt == 'adamax':
                optim = Adamax(lr=1)
            elif opt == 'adadelta':
                optim = Adadelta()
            elif opt == 'SGD':
                optim = SGD(lr=.01, momentum=.01, decay=.0001)
            elif opt == 'rmsprop':
                optim = RMSprop()
            else:
                optim = Adam(lr=.001)

            self.nn.compile(
                loss=loss,
                optimizer=optim,
                metrics=['accuracy', prec, re, f1]
            )

    def _create_model(self, S, A, H, lr, alpha=0.05, reg=0.01):
        inputs = Input(shape=self.shape)

        x = Conv2D(filters=H,
                   kernel_size=(8, 8),
                   strides=(4, 4),
                   use_bias=True,
                   bias_initializer=Constant(value=H),
                   padding='same',
                   kernel_regularizer=l2(reg)
                   )(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = MaxPool2D(pool_size=(2, 2),
                      strides=(2, 2))(x)

        x = Conv2D(filters=H * 2,
                   kernel_size=(4, 4),
                   strides=(2, 2),
                   use_bias=True,
                   bias_initializer=Constant(value=H * 2),
                   padding='same',
                   kernel_regularizer=l2(reg)
                   )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = MaxPool2D(pool_size=(2, 2),
                      strides=(2, 2))(x)

        x = Conv2D(filters=H,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   use_bias=True,
                   bias_initializer=Constant(value=H),
                   padding='same',
                   kernel_regularizer=l2(reg)
                   )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = MaxPool2D(pool_size=(2, 2),
                      strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(A, activation='linear', kernel_regularizer=l2(reg))(x)

        model = Model(inputs=inputs, outputs=x)

        return model


def timestamp(): return time.strftime('%Y-%m-%d@%H:%M:%S')
