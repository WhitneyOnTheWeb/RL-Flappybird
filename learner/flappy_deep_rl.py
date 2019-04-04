import sys
sys.path.append('../')
sys.path.append('../game')
sys.path.append('../saved')
sys.path.append('../images')
sys.path.append('/logs')

'---Python Extension Modules'
import os
import cv2
import csv
import math
import time
import json
import uuid
import arctic
import pickle
import imageio
import pymongo
import numpy as np
import pandas as pd
import pprint as pp
import random as rand
import matplotlib.pyplot as plt
import matplotlib.image as img
from IPython.display import SVG
from arctic import Arctic, CHUNK_STORE
from pymongo import MongoClient, WriteConcern

'---Keras / Tensorflow Modules'
import tensorflow as tf
from keras import metrics
from keras import backend as K
from keras.regularizers import l2
from keras.preprocessing import image
from keras.initializers import Constant
from keras.utils import plot_model, Sequence
from keras.applications import VGG16, ResNet50
from tensorflow import ConfigProto, Session, Graph
from keras.models import model_from_json, Sequential, Model
from keras.optimizers import SGD, Adam, Adamax, Adadelta, RMSprop
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, LeakyReLU
from keras.layers import Input, InputLayer, ReLU, Softmax, BatchNormalization

'---Keras Extension Modules'
import keras_metrics as km
from rl.core import Agent
from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent, AbstractDQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.agents.sarsa import SARSAAgent
from rl.memory import SequentialMemory, EpisodeParameterMemory, RingBuffer, Memory
from rl import policy as Policy
from rl.util import huber_loss


'''
Deep Reinforcement Learning: Flappy Bird with Keras
File:    flappy_deep_rl.py
Author:  Whitney King
Date:    March 31, 2019

References:
    Deep Learning Video Games 
        Author: Akshay Srivatsan
        github.com/asrivat1/DeepLearningVideoGames/
            blob/master/deep_q_network.py

    Using Deep-Q Network to Learn to Play Flappy Bird
        Author: Kevin Chen
        github.com/yenchenlin/DeepLearningFlappyBird

    Keras Reinforcement Learning
        Author: Matthias Plappert
        github.com/keras-rl/keras-rl

    Udacity ML Engineer Nanodegree Classroom
        tinyurl.com/yd7rye3w

    Visualizing Neural Network Activation
        Author: Arthur Juliani
        medium.com/@awjuliani/visualizing-neural-network-layer-activation-\
            tensorflow-tutorial-d45f8bf7bbc4

    How to Use Keras .fit() and .fit_generator()
        pyimagesearch.com/2018/12/24/how-to-use-keras-
            fit-and-fit_generator-a-hands-on-tutorial/


                    ▓▓▓▓▓▓▓▓▓▓▓▓
                ▓▓▓▓░░    ▓▓    ▓▓
              ▓▓░░░░    ▓▓        ▓▓
          ▓▓▓▓▓▓▓▓      ▓▓      ▓▓  ▓▓
        ▓▓        ▓▓    ▓▓      ▓▓  ▓▓
        ▓▓          ▓▓    ▓▓        ▓▓
        ▓▓░░      ░░▓▓      ▓▓▓▓▓▓▓▓▓▓▓▓
          ▓▓░░░░░░▓▓      ▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓
            ▓▓▓▓▓▓░░    ▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓
            ▓▓░░░░░░░░    ▓▓▒▒▒▒▒▒▒▒▒▒▓▓
              ▓▓▓▓░░░░░░    ▓▓▓▓▓▓▓▓▓▓
                  ▓▓▓▓▓▓▓▓▓▓        wk


" It may be hard for an egg to turn into a bird: 
  it would be a jolly sight harder for it to learn to fly
  while remaining an egg. We are like eggs at present. 
  
  And you cannot go on indefinitely being just an ordinary, 
  decent egg. We must be hatched or go bad. "

    ~ C. S. Lewis

'''

class RLAgent:
    def __init__(self, params, **kwargs):
        super(RLAgent, self).__init__(**kwargs)

        agent = params['agent']
        model = agent['model']

        if agent['name'] == 'DQN':
            self.worker = DQNAgent(
                model=model['network'],
                policy=model['policy'],
                test_policy=model['test_policy'],
                enable_double_dqn=model['double_dqn'],
                enable_dueling_network=model['dueling_network'],
                dueling_type=model['dueling_type'],
            )
        elif agent == 'SARSA':
            self.worker = SARSAAgent(
                model=model['network'],
                nb_actions=agent['action_size'],
                policy=model['policy'],
                test_policy=model['test_policy'],
                gamma=model['gamma'],
                nb_steps_warmup=model['warmup'],
                train_interval=model['train']['interval'],
            )
        elif agent == 'CEM':
            self.worker = CEMAgent(
                model=model['network'],
                nb_actions=agent['action_size'],
                memory=agent['memory']['store'],
                batch_size=model['train']['batch_size'],
                nb_steps_warmup=model['warmup'],
                train_interval=model['train']['interval'],
                elite_frac=model['alpha'],
                memory_interval=agent['memory']['interval'],
                theta_init=model['train']['initial_epsilon'],
                noise_decay_const=model['decay'],
                noise_ampl=model['noise_amp'],
            )
        # elif agent == 'DDPG':               #  stretch goal
        #     self.agent = DDPGAgent()
        else:
            self.worker = CustomDQN(
                nb_actions=agent['action_size'],
                memory=agent['memory']['store'],
            )


class CustomDQN(AbstractDQNAgent):
    def init(self, nb_actions, memory, **kwargs):
        super(CustomDQN, self).__init__(**kwargs)

    def run(self, params, env, util):
        agent = params['agent']
        game = env
        session = params['session']
        model = agent['model']
        save = model['save']
        memory = agent['memory']['store']
        train = model['train']
        episode = session['episode']
        step = episode['step']
        gif = episode['gif']
        image = step['image']
        settings = params['game']['settings']

        if not train['begin']:
            util.display_status('Beginning Warmup Period')
        else: 
            util.display_status('Training {} Keras Model'.\
                format(model['name']))
        session['start'] = time.time()
        for ep in range(1, agent['max_episodes'] + 1):
            '--- Reset parameters for each new episode'
            episode.update({ 
                'nb': ep,
                'log': [],
                'reward': 0,
                'score': 0,
                'wake': [1, 0],
                'gif': {
                    'x': [],
                    'xfm': [],
            }})
            step['t'] = 0
            t = step['t']

            '--- Send empty first action to game env, get output---'
            x, game_out = game.step(episode['wake'])
            util.update_nested_dict(settings, game_out)

            x_t = util.preprocess_input(x)
            step['state'] = util.create_state(x_t, params)

            '--- Begin stepping through game frame by frame'
            while session['status'] != 'exit' and t < episode['steps']:
                '--- Take the next step'
                session['step'] += 1
                t += 1
                
                '--- Determine action via random probability'
                if t % train['fpa'] == 0:
                    util.get_action(x_t, params)

                '--- Do next step and reshape as model input state'
                x1, game_out = game.step(step['action'])
                util.update_nested_dict(settings, game_out)

                x1_t = util.preprocess_input(x1)
                step['next'] = util.create_state(x1_t, params)
                step['terminal'] = settings['track']['crash']
                session['status'] = settings['track']['status']

                '--- Calculate reward for game action-state'
                util.get_reward(game, step)

                '--- Update episode / session parameter values'
                episode.update({
                    'score': game['track']['score'],
                    'reward': episode['reward'] + step['reward'],
                    'steps': step['t'],
                })
                '--- Update game screen capture image'                
                image.update({
                    'x': x1,
                    'xfm': x1_t,
                })
                gif['x'].append(image['x'])
                gif['xfm'].append(image['xfm'])

                '''---Store Experience in Memory---
                Tons of noise prior to the first pipe/gap. Curating 
                experiences in replay buffer helps prevent overwhemling
                the memory with states containing negative rewards.

                * limit stored experiences while observing random actions
                  * states past first 30 frames contain more entropy
                * only store new states past pipe[0]['x']
                '''
                if ep < 40 or t >= 50:
                    memory.append(util.get_replay(step, train))

                '--- Check for Game Over'
                if step['terminal']:                  # if terminal frame
                    util.game_over(params)    # trigger game over
                    continue                  # skip to next episode

                '--- Check if warming up or training'
                if session['step'] > train['warmup'] or train['begin']:
                    if not train['begin']:
                        util.display_status('Warmup Complete! Training Model')
                        train['begin'] = True
                    util.train_model(params)   # train using replay experience
                
                '--- Add step data to episode log'
                episode['log'].append(util.log_step(params))

                '''--- Check for CTRL+S/ESC to save model or 
                       save every save['interval'] steps'''
                if session['step'] % save['interval'] == 0 \
                    or session['step'] == train['warmup'] \
                    or session['status'] == 'save' or session['status'] == 'exit' \
                    or (ep == agent['max_episodes'] and step['terminal']):
                    util.save_model(params)

            '--- Check for ESC keypress to quit at end of episode'
            if session['status'] == 'exit': util.end_session(params)

        '--- Max episodes hit, end the active session'
        if session['status'] != 'exit': util.end_session(params)


class Buffer:
    def __init__(self, 
                 limit=50000,
                 mem_type = 'Sequential',
                 window_length=1):
        super(Buffer, self).__init__()
        if mem_type == 'RingBuffer':
            self.store = RingBuffer(limit)
        elif mem_type == 'EpisodeParameter':
            self.store = EpisodeParameterMemory(limit=limit, 
                                          window_length=window_length)
        else:
            self.store = SequentialMemory(limit=limit, 
                                          window_length=window_length)
        'methods:' # sample, append, get_config, nb_entries


class RLModel(Model):
    def __init__(self, S=4, A=2, H=64, lr=0.01,
                 alpha=0.05, reg=0.001, momentum=0.01,
                 decay=0.1, loss='logcosh', opt='adam',
                 model='custom', **kwargs):
        super(RLModel, self).__init__(**kwargs)
        self.shape = (80, 80, S)
        prec = km.binary_precision()
        re = km.binary_recall()
        f1 = km.binary_f1_score()

        if model == 'vgg16':
            self.nn = VGG16()
        if model == 'resnet50':
            self.nn = ResNet50()
        else:
            '''---Compile the model for use---'''
            self.nn = self._create_model(S, A, H, lr, alpha, reg)
            if opt == 'Adamax':
                optim = Adamax(lr=1)
            elif opt == 'Adadelta':
                optim = Adadelta()
            elif opt == 'SGD':
                optim = SGD(lr=.01, momentum=.01, decay=.0001)
            elif opt == 'RMSprop':
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