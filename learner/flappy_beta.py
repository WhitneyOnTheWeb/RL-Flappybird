import sys
sys.path.append('../')
sys.path.append('/game')
sys.path.append('/saved')
sys.path.append('/images')
sys.path.append('/logs')


'---Python Extension Modules'
import gc
import cv2
import csv
import math
import time
import json
import uuid
import pickle
import imageio
import numpy as np
import pandas as pd
import pprint as pp
import random as rand
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.image as img
from collections import defaultdict, Mapping
from flappy_util import Utility
from flappy_inputs import Inputs
from flappy_processor import FlappyProcessor
from flappy_callback import FlappySession
from game.flappy import Environment

'---Keras / Tensorflow Modules'
import tensorflow as tf
from rl.core import Agent
from keras import metrics
import keras_metrics as km
from keras import backend as K
from rl.util import huber_loss
from rl import policy as Policy
from keras.regularizers import l2
from rl.agents.cem import CEMAgent
from rl.agents.ddpg import DDPGAgent
from keras.preprocessing import image
from rl.agents.sarsa import SARSAAgent
from keras.initializers import Constant
from keras.utils import plot_model, Sequence
from keras.applications import VGG16, ResNet50
from tensorflow import ConfigProto, Session, Graph
from rl.agents.dqn import DQNAgent, AbstractDQNAgent
from keras.models import model_from_json, Sequential, Model
from keras.optimizers import SGD, Adam, Adamax, Adadelta, RMSprop
from rl.processors import MultiInputProcessor, WhiteningNormalizerProcessor
from rl.callbacks import ModelIntervalCheckpoint, Visualizer, FileLogger, Callback, CallbackList
from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger
from keras.callbacks import History, ModelCheckpoint, TensorBoard
from keras.callbacks import Callback as KerasCallback
from keras.callbacks import CallbackList as KerasCallbackList
from keras.layers import Input, InputLayer, ReLU, Softmax, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, LeakyReLU, Permute
from keras.layers import Input, InputLayer, ReLU, Softmax, BatchNormalization, Lambda
from rl.memory import SequentialMemory, EpisodeParameterMemory, RingBuffer, Memory
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy, Policy
from rl.policy import BoltzmannGumbelQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy
from rl.util import *

'''
Deep Reinforcement Learning for Flappy Bird
File:    agent.py
Author:  Whitney King
Date:    March 8, 2019

References:
    Deep Learning Video Games 
    Author: Akshay Srivatsan
    github.com/asrivat1/DeepLearningVideoGames/blob/master/deep_q_network.py

    Using Deep-Q Network to Learn to Play Flappy Bird
    Author: Kevin Chen
    github.com/yenchenlin/DeepLearningFlappyBird

    Udacity ML Engineer Nanodegree Classroom
    tinyurl.com/yd7rye3w

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
'''

'''Your mission, should you choose to accept it...'''

gpu = '/job:localhost/replica:0/task:0/device:GPU:0'
cpu = '/job:localhost/replica:0/task:0/device:CPU:0'

class Buffer(SequentialMemory):
    def __init__(self, 
                 limit=50000,
                 window_length=1):
        'methods:' # sample, append, get_config, nb_entries
        super(Buffer, self).__init__(
            limit=limit, 
            window_length=window_length
        )

class NeuralNet:
    def __init__(self,
                 sess,
                 S=4,
                 A=2,
                 H=64,
                 lr=0.01,
                 batch_size=32,
                 name='custom',
                 dueling_network=True,
                 dueling_type='max'):
        super(NeuralNet, self).__init__()
        #if name == 'vgg16':
        #    self.nn = VGG16()
        #if name == 'resnet50':
        #    self.nn = ResNet50()
        K.set_session(sess)
        self.shape = (S, 80, 80)
        
        with tf.device(gpu):
            inputs, outputs = self._create_model(S, A, H, lr)
            model = Model(inputs=inputs, outputs=outputs)

        if dueling_network:
            # get the second last layer of the model, abandon the last layer
            with tf.device(gpu):
                layer = model.layers[-2]
                nb_action = model.output._keras_shape[-1]
                # layer y has a shape (nb_action+1,)
                # y[:,0] represents V(s;theta)  y[:,1:] represents A(s,a;theta)
                y = Dense(nb_action + 1, activation='linear')(layer.output)
                # dueling_type == 'avg'
                # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))

                # dueling_type == 'max'
                # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))

                # dueling_type == 'naive'
                # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
                if dueling_type == 'avg':
                    outputs = Lambda(
                        lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - \
                                K.mean(a[:, 1:], axis=1, keepdims=True),
                        output_shape=(nb_action,)
                    )(y)
                elif dueling_type == 'max':
                    outputs = Lambda(
                        lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - \
                                K.max(a[:, 1:], axis=1, keepdims=True), 
                        output_shape=(nb_action,)
                    )(y)
                elif dueling_type == 'naive':
                    outputs = Lambda(
                        lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], 
                        output_shape=(nb_action,)
                    )(y)
                else:
                    assert False, "dueling_type must be one of {'avg','max','naive'}"
                    model = Model(inputs=inputs, outputs=outputs)
        self.nn = model

    def get_model(self): return self.nn

    def _create_model(self, S, A, H, lr, alpha=0.05, reg=0.01):
        inputs = Input(self.shape)
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
        x = Flatten()(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(4, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(4, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(4, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dense(A, activation='linear', kernel_regularizer=l2(reg))(x)

        return inputs, x

class BetaFlapDQN(DQNAgent):
    def __init__(self, inputs, buffer, sess_id, sess, **kwargs):
        self.util = Utility()
        self.sess = sess
        self.sess_id = sess_id

        game = inputs['game']
        agnt = inputs['agent']
        sess = agnt['session']
        eps = sess['episode']
        mod = inputs['model']
        trn = mod['training']
        sv = mod['save']
        mem = inputs['memory']

        '''---Environment Paramters---'''
        self.env_name = game['name']
        self.fps = game['fps']
        self.mode = game['difficulty']
        self.target = game['target']
        self.tick = game['tick']
        
        '''---Episode Parameters---'''
        self.nb_episodes = sess['max_ep']
        self.nb_max_episode_steps = game['fps'] * 60 * eps['max_time']
        self.nb_steps = self.nb_max_episode_steps * self.nb_episodes
        self.nb_steps_warmup = trn['warmup']
        self.nb_max_start_steps = trn['max_ep_observe']
        self.max_start_steps = trn['warmup']
        self.keep_gif_score = eps['keep_gif_score']
        
        '''---Agent / Model Parameters---'''
        self.name = agnt['name']
        self.nb_actions = agnt['action_size']
        self.delta_clip = agnt['delta_clip']

        self.training = trn['training']
        self.verbose = trn['verbose']
        self.lr = trn['learn_rate']
        self.eps = trn['initial_epsilon']
        self.value_max = trn['initial_epsilon']
        self.value_min = trn['terminal_epsilon']
        self.anneal = trn['anneal']
        self.shuffle = trn['shuffle']
        self.train_interval = trn['interval']
        self.validate = trn['validate']
        self.split = trn['split']
        self.action_repetition = trn['action_repetition']
        self.epochs = trn['epochs']
        self.epoch = 1

        prec = km.binary_precision()
        re = km.binary_recall()
        f1 = km.binary_f1_score()
        self.metrics = ['accuracy', 'mse', prec, re, f1]
        self.H = mod['filter_size']
        self.alpha = mod['alpha']
        self.gamma = mod['gamma']
        self.momentum = mod['momentum']
        self.decay = mod['decay']
        self.target_model_update = mod['target_update']
        self.type = mod['type']
        self.enable_double_dqn = mod['double_dqn']
        self.enable_dueling_network = mod['dueling_network']
        self.dueling_type = mod['dueling_type'] 
        
        self.limit = mem['limit']
        self.batch_size = mem['batch_size']
        self.window_length = mem['state_size']
        self.memory_interval = mem['interval']

        self.ftype = sv['ftype']
        
        self.vizualize = sv['visualize']
        self.save_full = sv['save_full']
        self.save_weights = sv['save_weights']
        self.save_json = sv['save_json']
        self.save_plot = sv['save_plot']
        self.save_interval = sv['save_n']
        self.log_interval = sv['log_n']
        self.saves = sv['save_path']
        self.save_path = self.util.get_save_dir_struct(
            self.saves,
            self.env_name
        )
        self.logs = sv['log_path']          
        self.util.display_status('Hyperparameters Successfully Loaded')
        
        '''Reference/Excerpt:  keras-rl DQN Atari Example
        https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
        # Select a policy. 
        # We use eps-greedy action selection, which means that a random action
        # is selected with probability eps. We anneal eps from init to term over 
        # the course of (anneal) steps. This is done so that the agent initially 
        # explores the environment (high eps) and then gradually sticks to 
        # what it knows (low eps). We also set a dedicated eps value that is 
        # used during testing. Note that we set it to 0.05 so that the agent 
        # still performs some random actions. 
        # This ensures that the agent cannot get stuck.
        # '''
        self.custom_model_objects = {
            'S': self.window_length,
            'A': self.nb_actions,
            'H': self.H,
            'lr': self.lr,
            'name': self.name,
            'batch_size': self.batch_size,
            'sess': self.sess,
            #dueling_network=self.enable_dueling_network,
            #dueling_type=self.dueling_type,
        }

        with tf.device(gpu):
            self.policy = LinearAnnealedPolicy(
                inner_policy=EpsGreedyQPolicy(
                    eps = self.value_max
                ), 
                attr='eps', 
                value_max=self.value_max,
                value_min=self.value_min, 
                value_test=self.alpha,
                nb_steps=self.anneal
            )
            self.test_policy = GreedyQPolicy()

            if mod['optimizer'].lower() == 'adamax':
                self.optimizer = Adamax(lr=self.lr)
            elif mod['optimizer'].lower() == 'adadelta':
                self.optimizer = Adadelta()
            elif mod['optimizer'].lower() == 'rmsprop':
                self.optimizer = RMSprop()
            elif mod['optimizer'].lower() == 'sgd':
                self.optimizer = SGD(
                    lr=self.lr, 
                    momentum=self.momentum, 
                    decay=self.decay,
            )
            else: self.optimizer = Adam(lr=self.lr)

        self.memory = buffer

    
        self.log_path =  self.util.get_log_dir_struct(
            self.sess_id, 
            self.logs, 
            self.ftype
        )

        self.util.display_status(
            'Keras GPU Session {} Beginning'.format(self.sess_id)
        )
          
        nn = NeuralNet(
            S=self.window_length,
            A=self.nb_actions,
            H=self.H,
            lr=self.lr,
            name=self.name,
            batch_size=self.batch_size,
            dueling_network=self.enable_dueling_network,
            dueling_type=self.dueling_type,
            sess=self.sess,
        )
        with tf.device(gpu):
            self.model = nn.get_model()

        self.util.display_status(
            '{} Keras Agent with {} Optimizer Built'.format(
                self.name, mod['optimizer']
        ))

        '''---Compile the model with chosen optimizer
        loss is calculated with lamba function based on model
        type selections (dueling, or double dqn)'''
        with tf.device(gpu):
            self.compile(   
                optimizer=self.optimizer,
                metrics=self.metrics,
            )

        self.util.display_status(
            '{} Agent Fully Initialized with Compiled Model'.format(self.name)
        )

        super(BetaFlapDQN, self).__init__(
            model=self.model,
            nb_actions=self.nb_actions,
            memory=self.memory,
            policy=self.policy,
            test_policy=self.test_policy,
            enable_double_dqn=self.enable_double_dqn,
            enable_dueling_network=self.enable_dueling_network,
            dueling_type=self.dueling_type,
            **kwargs
        )

    def load_saved_model_weights(self):
        try:
            self.model.load_weights('saved/FlappyBird_weights.h5')
            self.util.display_status('Saved Keras Model Weights Loaded')
        except:
            self.util.display_status('No Saved Keras Model Weights Found')

    def fit(self, iteration=1, max_iteration=1):
        self.load_saved_model_weights()

        with tf.device(gpu):
            self.env = Environment(
                target_score=self.target,
                difficulty=self.mode,
                fps=self.fps,
                tick=self.tick,
            )
        self.util.display_status(
            '{} Environment Emulation Initialized'.format(self.env_name)
        )

        if self.action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.\
                    format(self.action_repetition)
            )
        '''---Define Custom Callbacks and Processors BetaFlap'''
        FlappyCall = FlappySession()
        Flappy = FlappyProcessor()        

        '''---Flag Agent with as Training with on_train_begin()'''
        self._on_train_begin()
        FlappyCall.on_train_begin()
        
        self.training = True
        observation = None
        reward = None
        done = False
        info = None
        status = 'play'
        episode = np.int16(0)
        self.step = np.int16(0)
        action = np.int16(0)
        self.randQ = np.int16(0)
        self.reward = np.float16(0)
        idx = np.int16(0)
        flap = False
        episode_reward = None
        episode_score = None
        episode_step = None
        did_abort = False

        '''---Begin stepping through Episodes---'''
        # continue while global step is < max session steps
        while self.step < self.nb_steps:
            gc.collect()
            if observation is None:                 # new episode                   
                '''---Initialize Environment with No Action'''
                FlappyCall.on_episode_begin(episode)
                self.reset_states()    # reset all episode tracking parameters
                reward = None
                done = False
                info = {}
                action = None
                episode_step = np.int16(0)
                episode_score = np.int16(0)
                episode_reward = np.float32(0)

                wake = np.zeros([self.nb_actions]) # [0, 0]
                wake[0] = 1                        # [1, 0] --> don't flap
                o, r, done, info = self.env.step(wake)   # progress env 1 frame
                observation, r = Flappy.process_step(o, r, done, info)
                assert observation is not None

                '''---Each episode, begin with n random actions/steps'''
                if self.nb_max_start_steps == 0:
                    self.nb_random_start_steps = 0
                else: 
                    self.nb_random_start_steps = \
                    np.random.randint(self.nb_max_start_steps)
                '''---Perform random nb steps w/ rand action 
                      without adding them to experience replay memory'''
                for _ in range(self.nb_random_start_steps):
                    action = np.zeros([self.nb_actions])
                    randQ = rand.randrange(self.nb_actions)
                    action[randQ] = 1                   # flag selected action
                    o, r, done, info = self.env.step(action)  # progress env 1 frame
                    episode_step += 1
                
                    '''---Process output of randomized actions
                          without updating cumulative episode totals'''
                    observation = deepcopy(o)
                    observation, r = \
                        Flappy.process_step(observation, r, done, info)
                    if info['status'] =='exit': 
                        done = True
                        did_abort = True
                    if done: break          
                # warmup period complete
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None
                gc.collect()

            '''---Begin Iteratively Training Model Each Step
                * predict Q values / action (forward step)
                * use reward to improve the model (backward step)
            '''
            FlappyCall.on_step_begin(episode_step)

            '''---Predict Q Values Using Forward Method'''
            with tf.device(gpu):
                idx = self.forward(observation)
            action, flap = Flappy.process_action(idx, self.nb_actions)
            #episode_step += 1
            reward = np.float32(0)
            done = False
            for _ in range(self.action_repetition):
                o, r, d, i = self.env.step(action)
                observation = deepcopy(o)
                observation, r = Flappy.process_step(o, r, d, i)
                reward += r
                done = d
                info = i
                status = info['status']
                episode_step += 1
                if info['status'] =='exit': 
                    done = True
                    did_abort = True
                if done: break    # game over, end episode

            '''---Train the Model using Backward Method
            This function covers the bulk of the algorithm logic
                * store experience in memory
                * create experience batch, and predict Qs
                * train model on signle batch with selected optimizer
                * enable/disable double DQN or dueling network
                * update model target values
                * discount future reward and return model metrics
            '''
            with tf.device(gpu):
                metrics = self.backward(reward, terminal=done) 
            episode_reward += reward
            self.reward = episode_reward
            episode_score = info['score']
            

            '''---Log Step Data---'''
            step_log = {
                'step': episode_step,   # track episode step nb
                'episode': episode,
                'metrics': metrics,
                'flap': flap,
                'action': action,
                'reward': reward,
                'done': done,
                'training': self.training,
                'q_values': self.q_values,
                'info': info,
                'x': o,
                'x_t': observation,
            }
            FlappyCall.on_step_end(episode_step, step_log)
            gc.collect()

            #episode_step += 1
            self.step += 1

            if (self.step % self.save_interval) == 0 \
            or status == 'save': 
                self.save_model()
            if status =='exit': 
                done = True
                did_abort = True
            if self.nb_max_episode_steps and \
                (episode_step >= self.nb_max_episode_steps - 1):
                done = True       # max episode steps hit
            # We are in a terminal state but the agent hasn't yet seen it. 
            # perform one more forward-backward call and ignore the action
            if done:
                with tf.device(gpu):
                    self.forward(observation)
                    self.backward(0., terminal=False)
                episode_log = {
                    'sess_id': self.sess_id,
                    'episode': episode,
                    'reward': episode_reward,
                    'score': episode_score,
                    'steps': episode_step,   # track global step nb   
                    'gif': self.keep_gif_score,
                    'log_path': self.logs,
                    'iteration': iteration,
                }                        
                '''Episode Complete, Proceed to Next Iteration'''
                FlappyCall.on_episode_end(episode, episode_log)

                episode += 1          
                observation = None
                episode_step = None
                episode_reward = None
                episode_score = None
                gc.collect()
                
                if episode > self.nb_episodes or did_abort:
                    done = True       # max episode hit
                    break
            
        '''---Training Session Complete---'''   
        self.save_model()
        session_log = {
            'id': self.sess_id,
            'nb_steps': self.step,
            'did_abort': did_abort
        }
        FlappyCall.on_train_end(
            session_log, 
            self.sess_id, 
            self.log_path
        )
        self._on_train_end()  # end training session 
        if iteration >= max_iteration or did_abort: 
            self.env.close() 
            return True

    def forward(self, observation):
        # Select an action
        state = self.memory.get_recent_state(observation)
        with tf.device(gpu):
            self.q_values = self.compute_q_values(state)
        
        if self.training:  # LinearAnneal Greedy Epsilon
            with tf.device(gpu):
                action = self.policy.select_action(q_values=self.q_values)
        else:              #  GreedyQ
            with tf.device(gpu):
                action = self.test_policy.select_action(q_values=self.q_values)
        # Book-keeping for experience replay
        self.recent_observation = observation
        self.recent_action = action
        return action

    def backward(self, reward, terminal):
        '''Store latest step in experience replay tuple'''
        if self.step % self.memory_interval == 0 or self.reward > .011:
            if self.reward > .011: 
                self.util.display_status(
                    'Step {} Replay Experience Memory Saved'.format(self.step)
                )
            with tf.device(cpu):
                self.memory.append(
                        np.array(self.recent_observation), 
                        np.int16(self.recent_action), 
                        np.float32(reward), 
                        terminal,
                        training=self.training
                )
        metrics = []
        if not self.training:
            return metrics

        '''Begin Training on Batches of Stored Experiences'''
        if self.step > self.nb_steps_warmup \
        and self.step % self.train_interval == 0:
            with tf.device(gpu):
                batch = self.memory.sample(self.batch_size)
                assert len(batch) == self.batch_size
            
            state0_batch, reward_batch,action_batch, terminal1_batch, \
            state1_batch = \
                FlappyProcessor.process_state_batch(self, batch)

            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            '''Compute the Q-Values for Mini-Batch of Samples
            "Deep Reinforcement Learning with Double Q-learning"
            (van Hasselt et al., 2015):
            Double DQN: 
                - online network predicts actions
                - target network estimates Q values.
            '''
            if self.enable_double_dqn:
                with tf.device(gpu):
                    q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)
                # estimate Q values using the target network 
                # select maxQ value with the online model (computed above)
                with tf.device(gpu):
                    target_q_values = \
                    self.target_model.predict_on_batch(state1_batch)

                assert target_q_values.shape == \
                    (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            # Compute the q_values for state1, compute maxQ of each sample
            # prediction done on target_model as outlined in Mnih (2015),
            # it makes the algorithm is significantly more stable
            else:
                with tf.device(gpu):
                    target_q_values = \
                    self.target_model.predict_on_batch(state1_batch)

                assert target_q_values.shape == \
                    (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) 
            # update the affected output targets accordingly
            # Set discounted reward to zero for all states that were terminal
            discounted_reward_batch = self.gamma * q_batch
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
    
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(
                zip(targets, masks, Rs, action_batch)
            ):
                target[action] = R  # update with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            '''Train Using Sample Experience Batch'''
            # perform a single update on the entire batch
            # use a dummy target, as loss is computed complex Lambda layer
            # still useful to know the target to compute metrics properly
            if type(self.model.input) is not list:
                ins = [state0_batch] 
            else: state0_batch
            if self.validate: 
                split = self.split
            else: split = 0

            with tf.device(gpu):
                metrics = self.trainable_model.train_on_batch(
                    ins + [targets, masks], [dummy_targets, targets]
                )
                # THIS CAUSES A MEMORY LEAK IN CURRENT CONFIGURATION
                #metrics = self.trainable_model.fit(
                #    ins + [targets, masks], 
                #    [dummy_targets, targets],
                #    batch_size=None,
                #    epochs=self.epochs,
                #    verbose=self.verbose,
                #    validation_split=split,
                #    shuffle=self.shuffle
                #)
                gc.collect()
                
            # throw away individual losses
            if type(metrics) is list: 
                [m for idx, m in enumerate(metrics) if idx not in (1, 2)]
            else:
                metrics.history.update({'losses': self.policy.metrics})

        if self.target_model_update >= 1 \
        and self.step % self.target_model_update == 0:
            with tf.device(gpu):
                self.update_target_model_hard()
        return metrics
    
    
    def save_model(self):
        if self.save_full:
            '''---Save full model to single .h5 file---'''
            self.model.save(self.save_path + '_full.h5', overwrite=True)
            self.util.display_status(
                '{} Model Saved to {}'.format(
                    self.name, 
                    self.save_path + '_full.h5'
            ))
        if self.save_weights:
            '''---Save model weights to separate .h5 file---'''
            self.model.save_weights(self.save_path + '_weights.h5', overwrite=True)
            self.util.display_status(
                '{} Model Weights Saved to {}'.format(
                    self.name, self.save_path + '_weights.h5'
            ))
        if self.save_json:
            '''---Save model structure as JSON file---'''
            with open(self.save_path + '.json', 'a+') as f:
                json.dumps(self.model.to_json(), f)
            f.close()
            self.util.display_status(
                '{} Model Structure Saved to {}'.format(
                    self.name, self.save_path + '.json'
            ))
        if self.save_plot:
            plot_model(self.model, to_file=self.save_path + '_flow.png')
            self.util.display_status(
                '{} Neural Network Diagram Saved to {}'.format(
                    self.name, 
                    self.save_path + '_flow.png'
            ))