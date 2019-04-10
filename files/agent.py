import sys
sys.path.append('../')
sys.path.append('../game')

import os
import cv2
import csv
import time
import json
import uuid
import imageio
import numpy as np
import pprint as pp
import random as rand
import tensorflow as tf
import game.flappy as flappy
import matplotlib.image as img
import matplotlib.pyplot as plt
from IPython.display import SVG
from learner.deep_q import DeepQ
from learner.flappy_util import Utility
from learner.experience_replay import Buffer as replay
from collections import deque, namedtuple, defaultdict, Mapping, OrderedDict

'---Keras Extension Modules'
from keras import backend as K
import keras_metrics as km
from keras import metrics
from rl.core import Agent
from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent, AbstractDQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.agents.sarsa import SARSAAgent
from keras.utils import plot_model, Sequence
from tensorflow import ConfigProto, Session, Graph
from rl.memory import SequentialMemory, EpisodeParameterMemory, RingBuffer, Memory
from rl.policy import BoltzmannGumbelQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy, Policy
from rl.util import huber_loss
from rl.processors import MultiInputProcessor, WhiteningNormalizerProcessor
from rl.memory import SequentialMemory, EpisodeParameterMemory, RingBuffer, Memory


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

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class Parameters:
    def __init__(self, inputs):
        super(Parameters, self).__init__()

        '''---Initialize Neural Network---
        !!! All calls that begin with tf. should happen BEFORE and OUTSIDE 
            of any tf.Session or tf.InteractiveSession !!!

        Making these calls inside a session will cause computation graph to 
        grow each iteration, creating massive slow downs while training'''

        
        self.util.display_status('Beginning RL Parameter Initialization')

        self.alg = inputs['network']
        self.save = inputs['save_n']
        self.saves = inputs['save_path']
        self.logs = inputs['log_path']
        self.S = inputs['state_size']
        self.A = inputs['action_size']
        self.H = inputs['filter_size']
        self.lr = inputs['learn_rate']
        self.init_e = inputs['initial_epsilon']
        self.term_e = inputs['terminal_epsilon']
        self.observe = inputs['observe']
        self.anneal = inputs['anneal']
        self.gamma = inputs['gamma']
        self.ftype = inputs['ftype']

        '''---Initialize Game Emulation---'''
        self.k_observe = inputs['o_fpa']
        self.k_train = inputs['t_fpa']
        self.epochs = inputs['epochs']
        self.validate = inputs['validate']
        self.loss = inputs['loss_function']
        self.optimizer = inputs['optimizer']
        self.FPS = inputs['fps']
        self.episodes = inputs['max_games']
        self.gif_score = inputs['keep_gif_score']
        self.steps = self.FPS * inputs['max_time'] * 60
        self.mode = inputs['difficulty']
        self.target = inputs['target']
        self.tick = inputs['tick']
        self.util.display_status('Hyperparameters Successfully Loaded')
        self.name = inputs['name']
        

class Buffer(Memory):
    def __init__(self, 
                 limit=50000,
                 mem_type = 'Sequential',
                 window_length=1):
        super(Buffer, self).__init__(limit)
        if mem_type == 'RingBuffer':
            self.store = RingBuffer(limit)
        elif mem_type == 'EpisodeParameter':
            self.store = EpisodeParameterMemory(limit=limit, 
                                          window_length=window_length)
        else:
            self.store = SequentialMemory(limit=limit, 
                                          window_length=window_length)
        'methods:' # sample, append, get_config, nb_entries

class CustomDQN(DQNAgent):
    def __init__(self, model, nb_actions, memory,**kwargs):
        super(CustomDQN, self).__init__(
            processor=WhiteningNormalizerProcessor(),
            nb_actions=nb_actions, 
            memory=memory, 
            model=model,
            train_interval=1,
            memory_interval=1,
            target_model_update=10000,
            **kwargs)

        self.util = Utility()
        self.model = model
        self.memory = memory
        self.training = False
        self.step = 0

class RLAgent:
    def __init__(self, inputs):
        super(RLAgent, self).__init__()

        '''---Initialize Parameters with User Defined Settings---'''
        self.params = Parameters(inputs)
        self.inputs = inputs

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = Session(config=config, graph=tf.get_default_graph())

        '''---Session Tracking Parameters---'''
        self.G = 0       
        self.training = False          # total steps in session
        self.start = time.time()
        self.end = None
        self.elapsed = None
        self.log_episodes = []
        self.k = inputs['o_fpa']
        self.E = inputs['initial_epsilon']
        self.sess_id = self.params.util.get_id()
        K.set_session(self.sess)

        '''---Initialize Replay Memory---'''
        self.buff_size = inputs['limit']
        self.batch_size = inputs['batch_size']
        self.memory = Buffer(
            limit=inputs['limit'], 
            mem_type='Sequential', 
            window_length=inputs['state_size'],
        )

        self.env = flappy.Environment(
            target_score=inputs['target'],
            difficulty=inputs['difficulty'],
            fps=inputs['fps'],
            tick=inputs['tick'],
        )
        self.params.util.display_status('{} Environment Emulation Initialized'.\
            format(inputs['name']))

        self.params.util.display_status('Keras GPU Session {} Beginning'.\
            format(self.sess_id))

        self.model = DeepQ(
            S=inputs['state_size'],
            A=inputs['action_size'],
            H=inputs['filter_size'],
            lr=inputs['learn_rate'],
            loss=inputs['loss_function'],
            opt=inputs['optimizer'],
            model=inputs['model']
        )

        '''Reference:  keras-rl DQN Atari Example
        https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
        # Select a policy. We use eps-greedy action selection, which means that a random action is selected
        # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
        # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
        # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
        # so that the agent still performs some random actions. This ensures that the agent cannot get stuck. '''
        prec = km.binary_precision()
        re = km.binary_recall()
        f1 = km.binary_f1_score()

        self.policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(
                eps = inputs['initial_epsilon']
            ), 
            attr='eps', 
            value_max= inputs['initial_epsilon'],
            value_min=inputs['terminal_epsilon'], 
            value_test=inputs['alpha'],
            nb_steps=inputs['anneal']
        )
        
        self.dqn = CustomDQN(
            model=self.model.nn,
            nb_actions=inputs['action_size'],
            memory=self.memory.store,
            gamma=inputs['gamma'],
            nb_steps_warmup=inputs['observe'],
            train_interval=inputs['t_interval'],
            memory_interval=inputs['m_interval'],
            target_model_update=inputs['save_n'],
            batch_size=inputs['batch_size'],
            policy=self.policy,
            test_policy=GreedyQPolicy(),
            enable_double_dqn=inputs['double_dqn'],
            enable_dueling_network=inputs['dueling_network'],
            dueling_type=inputs['dueling_type'],
        )

        self.dqn.compile(
            optimizer=inputs['optimizer'],
            metrics=['accuracy', 'mse', prec, re, f1],
        )

        self.params.util.display_status('{} Keras Model Compiled'.\
            format(inputs['model']))

        '''---Load Saved Model / Weights---'''
        try:
            self.dqn.load_weights('saved\\FlappyBird_weights.h5')
            self.params.util.display_status('Saved Keras Model Weights Loaded')
        except:
            self.params.util.display_status('No Saved Keras Model Weights Found')
        self.params.util.display_status('{} Agent Fully Initialized'.\
            format(inputs['model']))
        print('-' * 80)

    '''---BetaFlap Agent for Deep Reinforcement Learing Gameplay'''
    def begin_work(self):
        p = self.params
        if not self.training:
            p.util.display_status('Beginning Warmup Period')
        else:
            p.util.display_status('Training {} Keras Model'.format(p.alg))

        '''---Begin Playing Game Episodes---'''
        self.status = 'play'
        self.start = time.time()
        for ep in range(1, p.episodes + 1):
            self.t = 0
            self.r_ep = 0
            self.s_ep = 0
            self.terminal = False
            self.frames = []                             # images of each frame
            self.t_frames = []
            self.log_steps = []
            self.nb = ep
            wake = np.zeros(p.A)
            wake[0] = 1                              # set action to none
            
            '''---Send Empty First Action to Emulator, Return State---'''
            x_t_c, r_0, terminal, game_log = self.env.step(wake)

            '''---Preprocess State for Model Input---
                * state is passed as a stack of four sequential frames
                * first step of episode creates stack of S frames
                * subsequent steps remove oldest frame, appends new frame'''
            self.x_t = self.preprocess(x_t_c, p)
            self.s_t = self.memory.store.get_recent_state(self.x_t)

            while self.status != 'exit' and self.t < p.steps:
                self.t += 1                 # limit episode frames
                self.G += 1   
                self.dqn.step += 1          # increment agent step
                a_t = np.zeros([p.A])
                Qs = np.zeros([p.A])
                randQ = rand.randrange(p.A)
                method = 'Wait'
                flap = False
                idx = 0

                '''---Pick Action: Best, Random, or None---
                    * add additional randomness to actions while observing
                    * having more variation here provides better training data
                    * Observation period occurs once, spans across episodes
                    * Fills replay memory with random training data
                    * After observation, training begings, epsilon anneals
                    * Determines Exploration or Exploitation probability
                    ---Follow Greedy Policy for max Q values---
                    ---Explore if rand <= Epsilon or Observing---
                '''
                Qs = self.dqn.compute_q_values(self.s_t)
                maxQ = self.dqn.forward(self.x_t)  # select action
                #idx = maxQ

                # use greedy Q policy to select action
                if self.t % self.k == 0:
                    if (rand.random() <= self.E) or (self.t <= p.observe):
                        '''Explore if rand <= Epsilon or Observing'''
                        method = 'Explore'               # always random if observing
                        idx = randQ
                    else:
                        '''---Follow Greedy Policy for maxQ values---'''
                        method = 'Exploit'  # prob of predicting Q goes up as E anneals
                        idx = maxQ
                        if self.E > p.term_e:
                            self.E = self.E - (p.init_e - p.term_e) / p.anneal

                a_t[idx] = 1       # flag action
                flap = False if a_t[0] else True
                #print(a_t)

                '''---Send Action, then proceed to training or next step---'''
                x_t1_c, r_t, terminal, game_log = self.env.step(a_t)
                self.terminal = terminal
                self.r_ep += r_t
                self.s_ep = game_log['score']
                self.status = game_log['status']
                hist = None
                                
                '''---Store Experience in Memory---
                Tons of noise prior to the first pipe/gap. Curating 
                experiences in replay buffer helps prevent overwhemling
                the memory with states containing negative rewards.

                * limit stored experiences while observing random actions
                  * states past first 30 frames contain more entropy
                * only store new states past pipe1_x
                '''
                #if ep < 30 or self.t > 49 and not self.training:
                #    self.memory.store.append(
                #        list(self.x_t), 
                #        list(a_t), 
                #        float(r_t), 
                #        bool(terminal), 
                #        training=True
                #    )

                '''---Check if Observing or Training---'''
                self.x_t1 = self.preprocess(x_t1_c, p)
                self.s_t1 = self.memory.store.get_recent_state(self.x_t1)

                '''This function covers the bulk of the algorithm logic
                    * store experience in memory
                    * create experience batch, and predict Qs
                    * train model using single stoastic gradient
                    * enable/disable double DQN or dueling network
                    * update model target values
                    * discount future reward and return model metrics
                '''

                hist = self.dqn.backward(r_t, terminal)
                if str(hist[0]) == 'nan':
                    hist = []
                print(hist, self.dqn.training, self.dqn.step)
                
                if self.G > p.observe or self.training:  # train when done observing
                    if not self.training:             # only trigger status change once
                        p.util.display_status('Warmup Complete! Training Model')
                        self.training = True          # set status to training
                        self.dqn.training = True
                        self.k = self.inputs['t_fpa']          # set frames per action to training rate

                    #e_bat = self.memory.store.sample(self.batch_size)
                    #s_bat = []
                    #r_bat = []
                    #a_bat = []
                    #s1_bat = []
                    #term_bat = []
                    #for e in e_bat:
                    #    s_bat.append(e[0])
                    #    a_bat.append(e[1])
                    #    r_bat.append(e[2])
                    #    s1_bat.append(e[3])
                    #    term_bat.append(0. if e[4] else 1.)
                    #s_bat = np.array(s_bat)
                    #s1_bat = np.array(s1_bat)
                    

                    #'''Fit the model with random batches from memory'''
                    #with self.sess.as_default():
                    #    Q_bat = self.dqn.compute_batch_q_values(s_bat)
                    #    Q_bat = self.model.nn.predict(s_bat)
                    #with self.sess.as_default():
                    #    tQ_bat = self.dqn.compute_batch_q_values(s1_bat)
                    #    tQ_bat = self.model.nn.predict(s1_bat)
                    #print('tQ_bat: ', tQ_bat)

                    #print('tQ_bat: ', tQ_bat)

                    #'''---Logic to Update maxQ Values greedily'''
                    #for i, x in enumerate(term_bat):
                    #    '''---tarQ_bat[i] = -1 if Terminal---'''
                    #    tQ_bat[i, [np.argmax(a_bat[i])]] = (-100) if x else \
                    #        r_bat[i] * p.gamma * np.max(Q_bat[i])

                    #with self.sess.as_default():
                        #hist = self.model.nn.fit(
                        #    x=s_bat, 
                        #    y=tQ_bat,
                        #    batch_size=self.inputs['batch_size'],
                        #    epochs=self.inputs['epochs'],
                        #    verbose=self.inputs['verbose'],
                        #    validation_split=self.inputs['split'],
                        #    shuffle=self.inputs['shuffle'],
                        #)

                '''---Log Step Data---'''
                data = {
                    'time': p.util.get_timestamp(),
                    'step': self.t,
                    'flap': flap,
                    'a_t': a_t,
                    'r_t1': r_t,
                    'term_t1': self.terminal,
                    'training': self.training,
                    'game_log': game_log,
                    'Qs': Qs[0],
                    'maxQ': maxQ,
                    'randQ': randQ,
                    'method': method,
                    'epsilon': self.E,
                    'metrics': hist,
                }
                self.log_steps.append(data)

                '''---Save Model Every (global_steps % n)---'''
                if (self.G % p.save) == 0 or (self.G == p.observe) \
                or (ep == p.episodes and self.terminal) \
                or self.status == 'save':
                    self.save_model(p)    # save model with CTRL+S

                '''---Check for Game Over---'''
                if self.terminal:               # if terminal frame
                    self.game_over(p)           # trigger game over
                    continue                    # skip to next episode

                '''---Proceed to Next Step/Transition/Frame---'''
                self.s_t = self.s_t1

            if self.status == 'exit':
                self.save_model(p)    # save and exit with ESC
                self.end_session(p)   # quit pygame and log session
                break
        if self.status != 'exit':          # max episodes hit, end session
            self.end_session(p)


    def log_episode(self, p):
        '''---Log episode information---'''
        data = {
            'nb': self.nb,
            'max_steps': p.steps,
            'reward': self.r_ep,
            'score': self.s_ep,
            'nb_steps': self.t,
            'steps': self.log_steps,
        }
        return data

    def log_session(self):
        '''---Log session information---'''
        data = {
            'id': self.sess_id,
            'start': self.start,
            'end': self.end,
            'elapsed': self.elapsed,
            'nb_steps': self.G,
            'episodes': self.log_episodes,
        }
        return data

    def write_log(self, data, p):
        name = '\\session' + '_' + self.sess_id
        path = os.getcwd() + '\\learner\\' + p.logs + name + p.ftype

        '--- Add data to log'
        p.util.display_status('Writing Session {} to Log File'.\
            format(self.sess_id))

        if p.ftype == '.json':  # dump data in json file 
            with open(path, 'a+') as f: 
                json.dump(data, f)
        else:                 # log to standard flat file
            with open(path, 'a+', newline='') as f:
                writer = csv.DictWriter(f, data.keys())
                if os.stat(path).st_size == 0: writer.writeheader()
                writer.writerow(data)
        f.close()
        p.util.display_status('Session Logged in {}'.format(path))
        #if ftype == 'mongodb': pass  # stretch goal, hook to flappy_freeze.py
    
    def save_model(self, p):
        path = p.saves + '/' + p.name
        '''---Save full model to single .h5 file---'''
        self.model.nn.save(
            path + '_full.h5', overwrite=True
        )
        p.util.display_status(
            '{} Model Saved to {}'.format(p.alg, path + '_full.h5')
        )
        plot_model(
            self.model.nn, to_file=path + '_flow.png'
        )
        p.util.display_status(
            '{} Neural Network Diagram Saved to {}'.format(
                p.alg, (path + '_flow.png')
        ))

        '''---Save model weights to separate .h5 file---'''
        self.dqn.save_weights(
            path + '_weights.h5', overwrite=True
        )
        p.util.display_status(
            '{} Model Weights Saved to {}'.format(
                p.alg, path + '_weights.h5'
        ))

        '''---Save model structure as JSON file---'''
        with open(path + '.json', 'a+') as f:
            json.dump(self.model.nn.to_json(), f)
        f.close()
        p.util.display_status(
            '{} Model Structure Saved to {}'.format(
                p.alg, path + '.json'
        ))
    
    def game_over(self, p):
        '''---Halt play, display stats, end current episode'''
        ep_stats = 'Game: {:<8}| Step: {:<10}| Rwd: {:<10.5f}| Score: {:<2}'.\
            format(self.nb, self.t, self.r_ep, self.s_ep)
        p.util.display_status(ep_stats)
        self.x_t1 = self.zeroed_observation(self.x_t1)    # clear next observation
        self.s_t1 = self.memory.store.get_recent_state(self.x_t1)
        self.log_episodes.append(self.log_steps)

        '''---Save GIF of Episodes with High Enough Scores---'''
        if self.s_ep >= p.gif_score:
            p.util.create_episode_gif(p)
        self.t = p.steps     # set as last step in episode

    def zeroed_observation(self, x):
        """Return an array of zeros with same shape as given observation
        Ref: keras-rl   memory.py
        """
        if hasattr(x, 'shape'):
            return np.zeros(x.shape)
        elif hasattr(x, '__iter__'):
            out = []
            for x in x:
                out.append(self.zeroed_observation(x))
            return out
        else: return 0.

    def end_session(self, p):
        '''---Exit PyGame and Close Session After Last Episode---'''
        p.util.display_status('Training Session Complete!')
        self.env.close()
        self.end = time.time()
        self.elapsed = time.gmtime(self.end - self.start)
        self.elapsed = time.strftime(
            '%H Hours %M Minutes %S Seconds', self.elapsed)
        sess_log = self.log_session()
        self.write_log(str(sess_log), p)    # track session information

        p.util.display_status('Elapsed Time: {}'.format(self.elapsed))
        print('  ___                   ___')
        print(' / __|__ _ _ __  ___   / _ \\__ ______ _')
        print('| (_ / _` | ''   \\/ -_) | (_) \\ V / -_) ''_|')
        print(' \\___\\__,_|_|_|_\\___|  \\___/ \\_/\\___|_|')
    
    #def create_state(self, x, p):
    #    if self.t == 0:
    #        x = np.stack(p.S * [x], axis=2)
    #    else:
    #        x = np.reshape(x, (1, 80, 80))  # channels dimension
    #        x = np.append(self.s_t[0, :, :, 1:], x, axis=2)
    #    x = np.reshape(x, (1, *x.shape))
    #    return x

    def preprocess(self, x, p):
        '''---Preprocess frames for neural network---
            * Reorient and resize: [512 x 288] -> [80 x 80] 
            * Convert from BGR to grayscale '''
        x_t = cv2.transpose(x)      # flips image from (Y, X) to (X, Y)
        self.frames.append(x_t)        # save image of frame
        x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)
        x_t = cv2.resize(x_t, (80, 80))
        x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)[1]
        self.t_frames.append(x_t)  # save transformed frame
        return x_t

    def create_episode_gif(self, p):
        '''---Save GIF of Episodes with High Enough Scores---'''
        p.util.display_status('Saving Episode {} as GIFs'.\
            format(self.nb))
        x_path = '/gifs/xep{}_{}.gif'.\
            format(self.nb, self.sess_id)
        xfm_path = p.log_path + '/gifs/xfmep{}_{}.gif'.\
            format(self.nb, self.sess_id)
        imageio.mimsave(x_path, self.frames)
        imageio.mimsave(xfm_path, self.t_frames)
        p.util.display_status('GIFs saved successfully')

    def plot_hist(self, hist):
        # Plot training & validation accuracy values
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()