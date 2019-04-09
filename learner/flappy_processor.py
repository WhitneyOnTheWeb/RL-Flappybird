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

    def process_reward(self, reward):
        gap_x, gap_y = False, False
        u1, u2, u3, u4 = False, False, False, False
        l1, l2, l3, l4 = False, False, False, False

        up = reward['pipes']['upper']
        low = reward['pipes']['lower']
        gaps = reward['pipes']['gaps']
        b = reward['player']

        '--- Reward player positioning if in gap between pipe corners'
        '--- Check player position against corners of pipes[0]'
        if b['right'] < up[0]['corners']['left'][0] \
                and b['y'] > up[0]['corners']['left'][1]:
            u1 = True  # player Sw of upper pipe[0] left corner
        if b['right'] < low[0]['corners']['left'][0] \
                and b['btm'] < low[0]['corners']['left'][1]:
            l1 = True  # player Nw of lower pipe[0] left corner
        '--- Check if player in right safe zone'
        if b['x'] > up[0]['corners']['right'][0] \
                and b['y'] > up[0]['corners']['right'][1]:
            u2 = True  # player SE of upper pipe[0] right corner
        if b['x'] > low[0]['corners']['right'][0] \
                and b['btm'] < low[0]['corners']['right'][1]:
            l2 = True
        '--- Check player position against corners of pipes[1]'
        '--- Check if player in left safe zone'
        if b['right'] < up[1]['corners']['left'][0] \
                and b['y'] > up[1]['corners']['left'][1]:
            u3 = True  # player Sw of upper pipe[1] left corner
        if b['right'] < low[1]['corners']['left'][0] \
                and b['btm'] < low[1]['corners']['left'][1]:
            l3 = True  # player Nw of lower pipe[1] left corner

        if up[0]['x_right'] < b['x_mid']:
            gap = gaps[1]     # past first pipes
            pipe = up[1]
        else:                 # first pipes
            gap = gaps[0]
            pipe = up[0]

        if gap['btm'] > b['btm'] > b['y'] > gap['top']:
            gap_y = True                  # level w/ pipe gap
            if pipe['x'] < b['x_mid'] < pipe['x_right']:
                gap_x = True              # in pipe gap

        '--- Reward scales up with steps/score/gap distance'
        step_dist = (1 + reward['step']) * (1 + reward['score'])
        tar_delta = reward['target'] - reward['score']

        tar_dist = np.sqrt(   # scaled reward booster
            np.abs((-reward['target'] ** 2 - reward['score'] ** 2) * 
            np.sqrt(step_dist * reward['step'])
        )) // (1 + tar_delta)

        gap_dist = np.sqrt(   # scaled reward booster
            np.abs((gap['mid'] - b['y_mid']) ** 2 - (up[0]['x_mid'] - b['x_mid']) ** 2) //
            step_dist
        ) // (tar_delta // 2)
        #print('tar: {}  st: {}  gap: {}  delta: {}'.\
        #    format(tar_dist, step_dist, gap_dist, tar_delta))

        award = tar_dist * gap_dist # base reward scales w/ dist/score
        if reward['step'] < 50:
            mul_penalty = (1 + tar_delta) / 5
        else: mul_penalty = 1
        self.msg = 'Danger zone!'
        award = (step_dist * tar_dist) - ((gap_dist + tar_delta) * mul_penalty) 
        if reward['step'] > 49:
            award += tar_dist ** gap_dist **2
        if reward['terminal']:
            award = -100
            self.msg = 'Boom!'
        elif reward['scored']:    # multiplier for scoring
            award += (
                ((tar_dist ** 2) * (gap_dist ** 2)) // tar_delta)
            self.msg = 'You scored!'
        elif gap_x and gap_y:  # player within pipe gap
            award += (
                tar_dist * step_dist // (1 + gap_dist * (1 + tar_delta ** 2))
            )
            self.msg = 'Great!'
        elif (u1 and l1) or (u2 and l2) or (u3 and l3):
            award += (np.sqrt(tar_dist + gap_dist) // tar_delta) * 10
            self.msg = 'Safe zone!!'

        '''Scale and constrain reward values, save as step reward'''
        '''---Hyperbolic Tangent of Reward---'''
        award = np.tanh(award * .000001)
        return award

    def process_action(self, action, nb_actions):
        '''Transform action from scalar index into binary array'''
        a_t = np.zeros([nb_actions])
        a_t[action] = 1       # flag action for env
        flap = False if a_t[0] else True
        return a_t, flap          # action index for experience tuple

    def process_step(self, o, r, d, i):
        observation = self.process_observation(o)
        reward = self.process_reward(r)
        return observation, reward
    
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