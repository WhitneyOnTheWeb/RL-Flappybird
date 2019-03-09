
'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    agent.py
Author:  Whitney King
Date:    March 8, 2019


References:
    Udacity ML Engineer Quadcopter Project
    github.com/WhitneyOnTheWeb/deep-learning-master/blob/master/Quadcopter/agents/agent.py
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

sys.path.append('game/')

class DeepQ():
    # RL Agent that learns using Deep-Q CNNs
    def __init__(self, task):
        self.task = task