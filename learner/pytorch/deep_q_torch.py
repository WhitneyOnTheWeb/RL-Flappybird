import sys
import os
import cv2
import math

import torch 
import torch.transforms as TN
import torch.transforms.functional as TF
import torch.nn.functional as NN
import torch.nn as nn
import torch.utils.data as data
import torchvision

import random as rand  
import numpy as np
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

class DeepQ(nn.Module):
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
    def __init__(self, lr = 0.01,
                 state_size = 4,
                 action_size = 2, 
                 hidden_size = 64,
                 name = 'DeepQ'):
        super(DeepQ, self).__init__()
        self.name = name
        self.lr   = lr
        self.S = state_size
        self.A = action_size
        self.H = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        '''---Create state size holding 4 frames---'''
        self.state   = torch.Tensor([None, 80, 80, self.S], 
                                    dtype = torch.float32)

        '''---Layer 1: Convolution w /ReLu Activation------'''
        self.conv1 = nn.Sequential( 
            # [80 x 80 x 4] -> [8, 8, 4, 64] -> [20 x 20 x 64] 
            nn.Conv2d(
                in_channels = self.S,
                out_channels = self.H,
                kernel_size = [8, 8, self.S, self.H],
                stride = 4),
            nn.ReLU(),                            # activation
            # [20 x 20 x 64] -> [1, 2, 2, 1] -> [10 x 10 x 64]
            nn.MaxPool2d(2))

        '''---Layer 2: Convolution w /ReLu Activation------''' 
        self.conv2 = nn.Sequential( 
            # [10 x 10 x 64] -> [4, 4, 64, 128] -> [5 x 5 x 128]  
            nn.Conv2d(
                in_channels = self.H,
                out_channels = self.H * 2,
                kernel_size = [4, 4, self.H, self.H * 2],
                stride = 2),
            nn.ReLU(),                          # activation
            # [5 x 5 x 128] -> [1, 2, 2, 1] -> [3 x 3 x 128]    
            nn.MaxPool2d(2))

        '''---Layer 3: Convolution w /ReLu Activation---''' 
        self.conv3 = nn.Sequential( 
            # [3 x 3 x 128] -> [3, 3, 128, 64] -> [3 x 3 x 64] 
            nn.Conv2d(
                in_channels = self.H * 2,
                out_channels = self.H,
                kernel_size = [3, 3, self.H * 2, self.H],
                stride = 1),
            nn.ReLU(),                        # activation
            # [3 x 3 x 64] -> [1, 2, 2, 1] -> [2 x 2 x 64] 
            nn.MaxPool2d(2))

        '''---Layer 4: Fully Connected Linear Output---''' 
        # [256, 1] -> [2, 1]
        conv_w = conv_size(conv_size(conv_size(self.state[1])))
        conv_h = conv_size(conv_size(conv_size(self.state[2])))
        size = conv_w * conv_h * 64
        self.out = nn.Linear(size, self.A)

        def conv_size(size, kernel_size = 4, stride = 1):
            '''calculate input dimensions before passing into linear output'''
            return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        fc = self.conv1(x)
        fc = self.conv2(fc)
        fc = self.conv3(fc)
        out = self.out(fc.view(fc.size(0), -1) )
        return x, out, fc