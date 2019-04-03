
'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    flappy_freeze.py
Author:  Whitney King
Date:    March 31, 2019

References:
    How to Use Keras .fit() and .fit_generator()
    pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
'''


import os
import sys
import time
import json
import uuid
import pickle
import random
import numpy as np
import pandas as pd
import pymongo
import arctic
from arctic import Arctic, CHUNK_STORE
from pymongo import MongoClient, WriteConcern
from keras.utils import Sequence
from collections import defaultdict


class StateGenerator(Sequence):
    def __init__(self, data, labels, batch_size=32,
                 dim=(80, 80), state_size=4,
                 action_size=2, channels=1, shuffle=True):
        self.x, self.y = data, labels
        self.dim = dim
        self.S = state_size
        self.A = action_size
        self.batch_size = batch_size
        self.channels = channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # needs the image repository to draw from
        # maybe use this to link in Arctic Store?
        pass

    def __data_generation(self, data_ids):
        'Generates data containing batch_size samples'
        pass

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class ReplayCollection:
    def __init__(self, session_id,
                 store='flappy',
                 episodes_to_keep=100000,
                 training_batch_size=32):

        self.store = Arctic('localhost')       # connect to Arctic DB
        self.max = episodes_to_keep

        if 'episodes' in self.store.list_libraries():
            self.idx = len(self.store['episodes'].list_symbols())
        else:
            self.store.initialize_library('episodes', lib_type=CHUNK_STORE)
            self.idx = 1

        if 'replay' not in self.store.list_libraries():
            self.store.initialize_library('replay', lib_type=CHUNK_STORE)

        self.episodes = self.store['episodes']
        self.replay = self.store['replay']

    def _replay_template(self, t, act, rwd, term, img):
        # This will go inside sess_ep symbol/ID
        # 'session_id': None, 'episode': None, 'step': None,
        data = {'step': None,
                'action': act,
                'reward': rwd,
                'terminal': term,
                'image': img}
        return data

        # This will go inside sess_ep symbol/ID 'idx': 1...
    def _episode_template(self, sess_ep): return {'sess_ep': sess_ep}

    def _sess_ep(self, sessid, ep): return sessid + '_' + ep

    def get_state(self, sessid, ep, t, S):
        state = np.array()
        reward = 0
        sess_ep = self._sess_ep(sessid, ep)
        read = self.replay.read(sess_ep)
        delta = S - t                 # state_size - ep_steps

        '''---Check for delta between state_size and steps---
             * Copy image[0] to fill delta
             * Fill rest of state with image[1]:image[t]'''
        if delta >= 0:                # state larger than steps taken
            for i in range(delta):    # fill delta with image[0]
                state.append(read['image'][0])
            for i in range(1, t):     # fill state with image[1]:image[t]
                state.append(read['image'][i])
                reward += read['reward'][i]  # reward unique images in state
            print('state.shape: ', state.shape)
        else:
            for i in range(t - S, t):
                state.append(read['image'][i])
                reward += read['reward'][i]  # reward all images in state

        action = read['action'][t]
        terminal = read['terminal'][t]

        return state, action, reward, terminal

    def get_experience_replay(self, sessid, ep, t, S):
        sess_ep = self._sess_ep(sessid, ep)
        read = self.replay.read(sess_ep)
        experience = ()

        '''Current State'''
        state, action, reward, terminal \
            = self.get_state(sessid, ep, t, S)

        '''Next State'''
        state_n, action_n, reward_n, terminal_n \
            = self.get_state(sessid, ep, t + 1, S)

        '''Create Experience Replay Tuple'''
        experience = (state, action, reward_n, state_n, terminal_n)
        return experience

    def get_random_sample(self, sessid, ep, t, S, batch_size):
        sess_ep = self._sess_ep(sessid, ep)
        ep_list = self.episodes.list_symbols()

        s_bat = []
        a_bat = []
        r_bat = []
        s1_bat = []
        term_bat = []

        # batch_size list of random episode index values
        idx = random.sample(range(len(ep_list)), batch_size)
        for i in idx:
            sample = self.replay.read(ep_list[i])
            steps = len(sample)
            rand_t = random.randint(0, steps - S)
            # get the experience replay tuple for episode[i]: step[rand_t]
            exp = self.get_experience_replay(sessid, ep, rand_t, S)
            s_bat.append(exp[0])
            a_bat.append(exp[1])
            r_bat.append(exp[2])
            s1_bat.append(exp[3])
            term_bat.append(exp[4])

        return s_bat, a_bat, r_bat, s1_bat, term_bat

    def add_step_to_episode(self, sessid, ep, t, act, rwd, term, img):
        '''---Populate Episode Step DataFrame, add to sess_ep---'''
        sess_ep = self._sess_ep(sessid, ep)
        ep_data = self._episode_template(sess_ep)
        re_data = self._replay_template(t, act, rwd, term, img)

        if t == 0:              # overwrite data on first step of each episode
            old = self.episodes.read(self.idx)
            self.episodes.write(self.idx, ep_data)  # save new sess_ep ID
            self.replay.delete(old['sess_ep'])     # delete old sess_ep replay
            self.replay.write(sess_ep, re_data)    # add first step replay data
            print('delete: ', old)
            print('replay: ', re_data)
        else:
            self.replay.append(self.idx, re_data)

        # check if max idx, and reset to 1 on game over
        if term:
            self.idx += 1
        elif self.idx % self.max == 0 and term:
            self.idx = 1   # max hit, reset idx
