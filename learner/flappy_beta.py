import sys
sys.path.append('../')
sys.path.append('../game')
sys.path.append('../saved')
sys.path.append('../images')
sys.path.append('/logs')

'---Python Extension Modules'
import gc
import os
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
import game.flappy as flappy
from flappy_util import Utility
from flappy_inputs import Inputs
from flappy_processor import FlappyProcessor
from flappy_callback import FlappySession

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

class Buffer(SequentialMemory):
    def __init__(self, 
                 limit=50000,
                 window_length=1):
        'methods:' # sample, append, get_config, nb_entries
        super(Buffer, self).__init__(
            limit=limit, 
            window_length=window_length
        )

class DeepQ:
    def __init__(self,
                 S=4,
                 A=2,
                 H=64,
                 lr=0.01,
                 batch_size=32,
                 model='custom'):
        super(DeepQ, self).__init__()
        self.shape = (S, 80, 80)
        #self.batch_size = batch_size
        if model == 'vgg16':
            self.nn = VGG16()
        if model == 'resnet50':
            self.nn = ResNet50()
        else: self.nn = self._create_model(S, A, H, lr)

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
        #x = MaxPool2D()(x)

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
        #x = MaxPool2D()(x)

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

        model = Model(inputs=inputs, outputs=x)
        return model

class BetaFlapDQN(DQNAgent):
    def __init__(self, inputs, **kwargs):
        self.util = Utility()

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

        '''---Session Parameters---'''
        self.sess_id = self.util.get_id()
        self.sess = self.config_session()
        
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
        self.log_path =  self.util.get_log_dir_struct(
            self.sess_id, 
            self.logs, 
            self.ftype
        )

        K.set_session(self.sess)
        self.util.display_status('Hyperparameters Successfully Loaded')

        self.memory = Buffer(
            limit=self.limit, 
            window_length=self.window_length,
        )
        self.util.display_status(
            'Built Replay Buffer Limited to {} States'.format(self.limit)
        )
        self.env = flappy.Environment(
            target_score=self.target,
            difficulty=self.mode,
            fps=self.fps,
            tick=self.tick,
        )
        self.util.display_status(
            '{} Environment Emulation Initialized'.format(self.env_name)
        )

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
        self.policy = LinearAnnealedPolicy(
            inner_policy=EpsGreedyQPolicy(
                eps = self.value_max
            ), 
            attr='eps', 
            value_max= self.value_max,
            value_min=self.value_min, 
            value_test=self.alpha,
            nb_steps=self.anneal
        )
        #self.policy = MaxBoltzmannQPolicy()
        self.test_policy = GreedyQPolicy()

        self.util.display_status(
            'Keras GPU Session {} Beginning'.format(self.sess_id)
        )
        self.custom_model_objects = {
            'S': self.window_length,
            'A': self.nb_actions,
            'H': self.H,
            'lr': self.lr,
            'model': self.name,
        }

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
            
        deepq = DeepQ(
            S=self.window_length,
            A=self.nb_actions,
            H=self.H,
            lr=self.lr,
            model=self.name,
            batch_size=self.batch_size,
        )
        self.model = deepq.get_model()

        self.util.display_status(
            '{} Keras Agent with {} Optimizer Built'.format(
                self.name, mod['optimizer']
        ))
        '''---Compile the model with chosen optimizer
           loss is calculated with lamba function based on model
           type selections (dueling, or double dqn)'''
        self.compile(   
            optimizer=self.optimizer,
            metrics=self.metrics,
        )
        self.util.display_status(
            '{} Agent Fully Initialized with Compiled Model'.format(self.name)
        )

        try:
            self.load_weights(os.getcwd() + '/saved/FlappyBird_weights.h5')
            self.util.display_status('Saved Keras Model Weights Loaded')
        except:
            self.util.display_status('No Saved Keras Model Weights Found')

        super(BetaFlapDQN, self).__init__(
            model=self.model,
            nb_actions=self.nb_actions,
            memory=self.memory,
            policy=self.policy,
            test_policy=self.test_policy,
            enable_double_dqn=self.enable_double_dqn,
            enable_dueling_network=self.enable_dueling_network,
            dueling_type=self.dueling_type,
            **kwargs)

    def fit(self, iteration):

        if self.action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.\
                    format(self.action_repetition)
            )

        '''---Define Custom Callbacks and Processors BetaFlap'''
        FlappyCall = FlappySession()
        Flappy = FlappyProcessor()
         # save every n steps
        ckpt = self.save_path + '_weights.h5'
        

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
            idx = self.forward(observation)

            action, flap = Flappy.process_action(idx, self.nb_actions)
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
            
            metrics = self.backward(reward, terminal=done) 
            episode_reward += reward
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

            episode_step += 1
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
                if did_abort: break   # save and exit with ESC key

                if episode > self.nb_episodes:
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
        self.env.close() 

    def forward(self, observation):
        # Select an action
        state = self.memory.get_recent_state(observation)
        with self.sess.as_default():
            self.q_values = self.compute_q_values(state)
        
        if self.training:  # LinearAnneal Greedy Epsilon
            action = self.policy.select_action(q_values=self.q_values)
        else:              # MaxBoltzmann GreedyQ
            action = self.test_policy.select_action(q_values=self.q_values)
        # Book-keeping for experience replay
        self.recent_observation = observation
        self.recent_action = action
        return action

    def backward(self, reward, terminal):
        '''Store latest step in experience replay tuple'''
        if self.step % self.memory_interval == 0:
            with self.sess.as_default():
                self.memory.append(
                    self.recent_observation, 
                    self.recent_action, 
                    reward, 
                    terminal,
                    training=self.training
                )
        metrics = []
        if not self.training:
            return metrics

        '''Begin Training on Batches of Stored Experiences'''
        if self.step > self.nb_steps_warmup \
        and self.step % self.train_interval == 0:
            with self.sess.as_default():
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
                with self.sess.as_default():
                    self.q_values = self.model.predict_on_batch(state1_batch)
                assert self.q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(self.q_values, axis=1)
                assert actions.shape == (self.batch_size,)
                # estimate Q values using the target network 
                # select maxQ value with the online model (computed above)
                with self.sess.as_default():
                    self.target_q_values = \
                        self.target_model.predict_on_batch(state1_batch)

                assert self.target_q_values.shape == \
                    (self.batch_size, self.nb_actions)
                self.q_batch = self.target_q_values[range(self.batch_size), actions]
            # Compute the q_values for state1, compute maxQ of each sample
            # prediction done on target_model as outlined in Mnih (2015),
            # it makes the algorithm is significantly more stable
            else:
                with self.sess.as_default():
                    self.target_q_values = \
                        self.target_model.predict_on_batch(state1_batch)

                assert self.target_q_values.shape == \
                    (self.batch_size, self.nb_actions)
                self.q_batch = np.max(self.target_q_values, axis=1).flatten()
            assert self.q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) 
            # update the affected output targets accordingly
            # Set discounted reward to zero for all states that were terminal
            discounted_reward_batch = self.gamma * self.q_batch
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
            if self.validate: split = self.split
            else: split = 0

            metrics = self.trainable_model.fit(
                ins + [targets, masks], 
                [dummy_targets, targets],
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                validation_split=split,
                shuffle=self.shuffle,
            )
            #metrics = self.trainable_model.train_on_batch(
            #    ins + [targets, masks], [dummy_targets, targets])
            # throw away individual losses
            #metrics = \
            #    [m for idx, m in enumerate(metrics) if idx not in (1, 2)]
            #metrics += self.policy.metrics

        if self.target_model_update >= 1 \
        and self.step % self.target_model_update == 0:
            self.update_target_model_hard()
        return metrics
    
    def config_session(self):
        config = ConfigProto(
            device_count = {'GPU': 2},
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
        )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        
        graph = tf.get_default_graph()
        sess = Session(config=config, graph=graph)
        return sess
    
    
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