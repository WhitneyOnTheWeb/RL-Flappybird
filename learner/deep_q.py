'''
Deep-Q Reinforcement Learning for Flappy Bird
Author:  Whitney King
Date:    March 8, 2019

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

import os
import sys
import cv2
import opencv
import tensorflow as tf
import random as rand  
import numpy as np
import flappy
from collections import deque

sys.path.append('../game/')

GAME         = 'flappy'  # identifer for game in log files
ACTIONS      = 2         # number of possible actions
GAMMA        = .99       # decay rate of past observations
OBSERVE      = 500.      # timesteps to observe before training
EXPLORE      = 500.      # number frames to anneal epsilon over
TERM_EPSILON = 0.05      # final value of epsilon
INIT_EPSILON = 1.0       # inital value of epsilon
REPLAY_MEM   = 50000     # number of transitions to remember
BATCH        = 32        # minibatch size
K            = 1         # number of frames per action

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape = shape))

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, 
                        strides = [1, stride, stride, 1], 
                        padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, 
                          ksize = [1, 2, 2, 1], 
                          strides = [1, 2, 2, 1], 
                          padding = "SAME")

def QNetwork():
    # Network Weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])