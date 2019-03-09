'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    task.py
Author:  Whitney King
Date:    March 8, 2019

References:
    Udacity ML Engineer Quadcopter Project
    github.com/WhitneyOnTheWeb/deep-learning-master/blob/master/Quadcopter/tasks/takeoff.py

    Udacity ML Engineer Nanodegree Classroom
    tinyurl.com/yd7rye3w
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



# Your mission, should you choose to accept it...
class FlappyBird():
    # Defines the goal and provides feedback to the agent
    def __init__(self,):
        # Initialize object to play Flappy Bird
        self.GAME          = 'flappy'  # identifier for log files
        self.ACTIONS       = 2         # number of possible actions
        self.GAMMA         = .99       # decay rate of observations
        self.OBSERVE       = 500.      # steps to observe before training
        self.EXPLORE       = 500.      # number frames to anneal over epsilon
        self.TERM_EPSILON  = 0.05      # terminal value of epsilon
        self.INIT_EPSILON  = 1.0       # initial value of epsilon
        self.REPLAY_MEMORY = 50000     # number of transistion to track
        self.BATCH         = 32        # minibatch size
        self.K             = 1         # number of frames per action
        self.TARGET_SCORE  = 40        # goal; defines the line for success

#---------------------------------------------------------
# Method: play_game()

# Establishes a penalty and reward structure for take off
# based on position and angle of quadcopter
#---------------------------------------------------------
    def play_game(self):
        return
    
    def cost(self, readout):
        # Defines the cost functions of actions for state
        a = tf.placeholder('float', [None, self.ACTIONS])
        y = tf.placeholder('float', {None})
        readout_action = tf.reduce_sum(tf.mul(readout, a), 
                                       reduction_indicies = 1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        return step


#---------------------------------------------------------
# Method: get_reward()

# Establishes a penalty and reward structure for proximity
# pf bird spirte to pipe sprites. Reward maximizing
# the distance, and penalize the closer bird is to a pipe
#---------------------------------------------------------
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0                       # rewards maximizing distance
        penalty = 0                      # penalizes getting too close to pipes
        
        #current_pos = self.sim.pose[:3]  # position has moved from the start
        #pos_dist = np.array(current_pos) - np.array(self.target_pos) # distance between current and target pos
        #e_angles = self.sim.pose[3:6]    # euler angle of each axis
        #dist = abs(pos_dist).sum()
        
        # add a penalty for euler angles at take off to steady lift
        #penalty += abs(e_angles).sum() ** 2
        
        # add a penalty for distance from target
        #penalty += dist
        
        # add reward for nearing or reaching target goal
        #if dist == 0: reward += 1000     # very large reward for reaching target
        #elif dist <= 10: reward += 500   
        #elif dist <= 50: reward += 250   # increase reward as dist gap closes
        #elif dist <= 100: reward += 100
        #else: reward += 10               # small reward for episode completion
            
#return np.tanh(reward - penalty *.005) # deduct penalties from final reward