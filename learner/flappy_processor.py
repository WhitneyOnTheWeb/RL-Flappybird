import os
import sys
import csv
import cv2
import time
import json
import pickle
import numpy as np
import pprint as pp
import random as rand
import matplotlib.image as img
import matplotlib.pyplot as plt
from rl.core import Processor


class FlappyProcessor(Processor):
    def process_observation(self, observation):
        '''---Preprocess frames for neural network---
            * Reorient and resize: [512 x 288] -> [80 x 80] 
            * Convert from BGR to grayscale '''
        x_t = cv2.transpose(observation)      # flips image from (Y, X) to (X, Y)
        x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)
        x_t = cv2.resize(x_t, (80, 80))
        x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)[1]
        return x_t

    def process_step(self, o, r, d, i):
        observation = self.process_observation(o)
        reward = self.process_reward(r)
        return observation, reward
    
    def process_reward(self, reward):
        '--- Reward scales up with steps/score/gap distance'
        step_dist = (1 + reward['step']) * (1 + reward['score'])
        tar_delta = reward['target'] - reward['score']

        tar_dist = np.sqrt(   # scaled reward booster
            np.abs((-reward['target'] ** 2 - reward['score'])) * 
            np.sqrt(step_dist * reward['step'])) // (1 + tar_delta)

        if reward['step'] < 50:
            mul_penalty = np.sqrt(tar_delta)
        else: mul_penalty = 1
        self.msg = 'Danger zone!'

        award = (tar_dist ** 4) / mul_penalty  # base reward scales w/ dist/score
        if reward['step'] > 49:
            award += 1
        else: award = award / ((1 + tar_delta) * mul_penalty)
        if reward['terminal']:
            award = -100
            self.msg = 'Boom!'
        elif reward['scored']:    # multiplier for scoring
            award += (
                np.sqrt((1 + tar_dist ** 2) * (1 + step_dist ** 2)))
            self.msg = 'You scored!'

        #print('step: {} | tar: {} | delta: {}'.format(step_dist, tar_dist, tar_delta))

        '''Scale and constrain reward values, save as step reward'''
        '''---Hyperbolic Tangent of Reward---'''
        award = np.tanh(award * .00001)
        return award

    def process_action(self, action, nb_actions):
        '''Transform action from scalar index into binary array'''
        a_t = np.zeros([nb_actions])
        a_t[action] = 1       # flag action for env
        flap = False if a_t[0] else True
        return a_t, flap          # action index for experience tuple

    
    def process_state_batch(self, batch):
        # unpack experiences into individual lists
        s0 = []
        r = []
        a = []
        t = []
        s1 = []
        for e in batch:
            s0.append(e.state0)
            s1.append(e.state1)
            r.append(e.reward)
            a.append(e.action)
            t.append(0. if e.terminal1 else 1.)
        '''May need to ensure this works correctly, or if preprocessing
        needs to be adjusted when fed into memory and the model
        '''
        s0 = np.array(s0)
        s1 = np.array(s1)
        t = np.array(t)
        r = np.array(r)

        return s0, r, a, t, s1