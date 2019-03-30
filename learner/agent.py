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
import pprint as pp
import tensorflow as tf
import random as rand  
import numpy as np
import game.flappy as flappy
import matplotlib.pyplot as plt
import matplotlib.image as img

import learner.flappy_mongo as db
from learner.experience_replay import Buffer
from learner.deep_q import DeepQ
from collections import deque
from keras import backend as K

'''
Deep-Q Reinforcement Learning for Flappy Bird
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
class Parameters:
    def __init__(self,
                 state_size = 4,           
                 action_size = 2,            
                 hidden_size = 64,           
                 learn_rate = 0.001,         
                 algorithm = 'DeepQ',             # neural network ID
                 save_path = 'saved',            # model save path
                 log_path = 'learner/logs/',      # session log path
                 states_in_memory = 50000,        # replay memory size
                 training_sample_size = 32,       # minibatch
                 frames_per_action = 1,
                 max_games = 10000,               # number times to play game
                 game_difficulty = 'hard',        # easy, medium, or hard
                 fps = 30,                        # frames per second
                 keep_gif_for_score = 5,          # episode score to save GIF
                 max_game_minutes = 10,
                 game_score_target = 40,
                 save_every_n_steps = 20000,
                 training = False,                # observe then train if False   
                 reward_discount_factor = 0.99,
                 initial_epsilon = 0.5,
                 terminal_epsilon = 0.0001, 
                 observation_steps = 10000,       # pre training
                 exploration_steps = 1000000):    # epsilon decay steps 
        super(Parameters, self).__init__()   

        '''---Initialize Neural Network---
        !!! All calls that begin with tf. should happen BEFORE and OUTSIDE 
            of any tf.Session or tf.InteractiveSession !!!

        Making these calls inside a session will cause computation graph to 
        grow each iteration, creating massive slow downs while training'''
        print('{} | Initializing Neural Network...'.format(timestamp()))
        self.alg      = algorithm
        self.save     = save_every_n_steps
        self.saves    = save_path
        self.logs     = log_path
        self.S        = state_size
        self.A        = action_size
        self.H        = hidden_size       
        self.lr       = learn_rate
        self.E        = initial_epsilon
        self.init_e   = initial_epsilon
        self.term_e   = terminal_epsilon
        self.observe  = observation_steps
        self.anneal   = exploration_steps
        self.gamma    = reward_discount_factor
        self.training = training                           
        self.model    = DeepQ(self.S, self.A, self.H, self.lr)         
        self.graph    = tf.get_default_graph()    # tensor graph
        print('{} | {} Model Successfully Compiled...'.\
            format(timestamp(), self.alg))

        
        '''---Initialize Game Emulation---'''
        self.k         = frames_per_action
        self.FPS       = fps      
        self.episodes  = max_games  
        self.gif_score = keep_gif_for_score
        self.steps     = self.FPS * max_game_minutes * 60
        self.status    = 'play'
        self.mode      = game_difficulty
        self.target    = game_score_target
        self.game      = flappy.GameState(self.target, 
                                          self.mode, 
                                          self.FPS)
        self.name      = self.game.name
        print('{} | {} Emulation Initialized...'.\
              format(timestamp(), self.name))


        '''---Initialize Replay Memory---'''
        self.buffer   = states_in_memory
        self.batch    = training_sample_size
        self.replay   = db.ReplayCollection(states_to_keep = self.buffer,
                                            training_batch_size = self.batch)
        self.memory   = Buffer(self.buffer, self.batch, self.replay)  # replay memory buffer

        '''---Session Tracking Parameters---'''
        self.R        = []                    # rewards across episodes
        self.Sc       = []                    # score across episodes
        self.T        = []                    # steps across episodes
        self.G        = 0                     # total steps in session
        self.start    = None


        '''---Episode Tracking Parameters---'''
        self.ep_id    = uuid.uuid1()          # unique episode ID
        self.r_ep     = 0                     # rewards per episode
        self.s_ep     = 0                     # game score of episode
        self.t        = 0                     # number of steps in episode
        self.frames   = []                    # images of each frame
        self.t_frames = []                    # images of transformed frame
        
        
        '''---Step Tracking Parameters---'''
        self.s_t      = None                  # current state
        self.a_t      = np.zeros([self.A])         # action
        self.r_t      = None                  # reward for action-state
        self.s_t1     = None                  # next state
        self.x_t_c    = None                  # game frame as image
        self.x_t      = None                  # transformed game frame
        self.x_t1_c   = None                  # next frame as image
        self.x_t1     = None                  # transformed next frame
        self.terminal = False                 # game over
        self.msg      = None                  # human-friendly helpful output
        

        '''---Action Selection Parameters---'''
        self.Qs       = [None, None]          # reward for each possible action
        self.randQ    = None                  # select a random action index
        self.maxQ     = None                  # select action index via max Q-value
        self.meth     = 'Wait'                # Explore, Exploit, or Wait
        self.idx      = 0                     # action index to flag
        self.flap     = False                 # track if agent flapped


class Agent:
    def __init__(self,
                 state_size = 4,           
                 action_size = 2,            
                 hidden_size = 64,           
                 learn_rate = 0.001,         
                 algorithm = 'DeepQ',
                 save_path = 'saved',
                 log_path = 'learner/logs/',
                 states_in_memory = 50000,
                 training_sample_size = 32,
                 frames_per_action = 1,
                 max_games = 10000,
                 game_difficulty = 'hard',
                 fps = 30,
                 keep_gif_for_score = 5,
                 max_game_minutes = 10,
                 game_score_target = 40,
                 save_every_n_steps = 20000,
                 training = False,                
                 reward_discount_factor = 0.99,
                 initial_epsilon = 0.5,
                 terminal_epsilon = 0.0001, 
                 observation_steps = 10000,
                 exploration_steps = 1000000): 
        super(Agent, self).__init__()


        '''---Initialize Parameters with User Defined Settings---'''
        self.params = Parameters(state_size, action_size, hidden_size, 
                                 learn_rate, algorithm, save_path, 
                                 log_path, states_in_memory, 
                                 training_sample_size, frames_per_action,
                                 max_games, game_difficulty, fps, 
                                 keep_gif_for_score, max_game_minutes, 
                                 game_score_target, save_every_n_steps, 
                                 training, reward_discount_factor, 
                                 initial_epsilon,  terminal_epsilon, 
                                 observation_steps, exploration_steps)

        '''---Initialize Session and Variables---'''
        self.sess_id  = time.strftime('%Y%m%d%H%M%S')   # unique ID
        self.s_log    = '{}session_{}.csv'.\
            format(log_path, self.sess_id)
        self.config   = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess     = tf.Session(config = self.config)
        K.set_session(self.sess)                             # THIS MAY NOT GO HERE - NEEDS THOUROUGH TESTING 
        print('{} | Keras Session Initialized (ID: {})...'.\
        format(timestamp(), self.sess_id))

        '''---Load Saved Model / Weights---'''
        try:
            self.params.model.nn.load_weights('saved\\FlappyBird_model.h5')
            print('{} | Stored Model Weights Loaded Successfully...'.\
            format(timestamp()))
        except:
            print('{} | No Saved Model Weights to Load...'.\
            format(timestamp()))

        # model weights loaded as first priority since we have the model
        # this can be used if an entire model needs to be loaded
        #try:
        #    self.load_model(self.params)
        #except:
        #    print('{} | No Saved Models to Load...'.\
        #    format(timestamp()))

    '''---BetaFlap Agent for Deep Reinforcement Learing Gameplay'''
    def play(self): 
        p = self.params
        p.start = time.time()
        if not p.training:                                     
            print('{} | Begining Observation Period...'.\
            format(timestamp()))
        else:
            print('{} | Training Session Beginning...'.\
            format(timestamp()))

        '''---Begin Playing Game Episodes---'''
        for ep in range(0, p.episodes):         
            p.ep_id        = uuid.uuid1()                   # unique episode ID
            p.r_ep         = 0
            p.t            = 0                   
            p.frames       = []                             # images of each frame
            p.trans_frames = []              
            wake           = np.zeros(p.A)
            wake[0]        = 1                              # set action to none

            '''---Send Empty First Action to Emulator, Return State---'''
            p.x_t_c, r_0, p.r_ep, p.s_ep, p.terminal, p.msg, p.status = p.game.step(wake)

            '''---Preprocess State for Model Input---
                * state is passed as a stack of four sequential frames
                * first step of episode creates stack of S frames
                * subsequent steps remove oldest frame, appends new frame'''
            p.x_t = self.preprocess(p.x_t_c, p)
            p.s_t = np.stack([p.x_t] * p.S, axis = 2)
            p.s_t = np.reshape(p.s_t, (1, *p.s_t.shape))     #  [1, 80, 80, p.S]
                        
            while p.status != 'exit' and p.t < p.steps:               
                p.t    += 1                    # limit episode frames
                p.G    += 1                    # increment session frames
                p.a_t   = np.zeros([p.A])      # reset to no action
                p.Qs    = [None, None]
                p.randQ = None
                p.maxQ    = None
                p.meth  = 'Wait'
                p.idx   = 0                                 

                '''---Pick Action: Best, Random, or None---'''          
                if p.t % p.k == 0:               # frames per action
                    self.action(p.s_t, p)             
                p.a_t[p.idx] = 1                 # flag action index
                p.flap = False if p.a_t[0] else True 
                
                '''---Send Action, then Preprocess Next State---'''
                p.x_t1_c, p.r_t, p.r_ep, p.s_ep, p.terminal, p.msg, p.status = p.game.step(p.a_t)

                p.x_t1 = self.preprocess(p.x_t1_c, p)
                p.x_t1 = np.reshape(p.x_t1, (80, 80, 1))  # channels dimension
                p.s_t1 = np.append(p.x_t1,p.s_t[0, :, :, :p.S-1], axis = 2)
                p.s_t1 = np.reshape(p.s_t1, (1, *p.s_t1.shape))
                #print('Step: {}  |  {}  |  Flap: {}  |  Reward: {:.3f}  |  {}'.\
                #      format(t, meth, flap, r_t, msg))

                '''---Store Experience in Memory---'''
                p.memory.add(p.s_t, p.a_t, p.r_t, p.s_t1, p.terminal)
                #p.replay.add(p.s_t, p.a_t, p.r_t, p.s_t1, p.terminal)
                p.r_ep += p.r_t                    # update total episode rewards

                '''---Check for Game Over---''' 
                if p.terminal:                     # if terminal frame
                    self.game_over(ep, p)          # trigger game over
                    continue                       # skip to next episode
     
                '''---Check if Observing or Training---'''
                if p.G > p.observe or p.training:    # train when done observing
                    if not p.training:           # only trigger status change once
                        print('{} | Observations Complete; Training Model...'.\
                        format(timestamp()))
                        p.training = True       # set status to training

                    '''---Train Model with Experience Replay from Memory---'''
                    self.replay(p)

                '''---Log Step Data---'''
                self.log_state(ep, p)

                '''---Check Keydown Status of Game Emulator---'''
                if p.status == 'save':
                    self.save_model(p)    # save model with CTRL+S

                '''---Save Model Every (global_steps % n)---'''
                if p.G % p.save == 0 \
                    or p.G == p.observe \
                    or (ep == p.episodes and p.terminal):
                    self.save_model(p)

                '''---Proceed to Next Step/Transition/Frame---'''
                p.s_t = p.s_t1

            if p.status == 'exit':
                p.episodes = ep + 1   # flag as last episode
                self.save_model(p)    # save and exit with ESC
                self.end_session(p)   # quit pygame and log session
                break

        if p.status != 'exit': self.end_session(p)

    def end_session(self, p):
        '''---Exit PyGame and Close Session After Last Episode---'''    
        print('\n{} | Training Session Complete!'.format(timestamp()))             
        p.game.quit_game()                # quit pygame after last episode
        print('\n{} | Storing Memory in Replay DB...'.format(timestamp()))
        self.store_experience(p)               
        print('\n{} | Replay DB Successfully Updated...'.format(timestamp()))

        end = time.time()
        elapsed = time.gmtime(end - p.start)
        elapsed = time.strftime('%H Hours %M Minutes %S Seconds', 
                                elapsed)
        self.log_session(p, elapsed)     # track session information
        print('{} | Elapsed Time: {}'.format(timestamp(), (elapsed)))
        print('  ___                   ___')         
        print(' / __|__ _ _ __  ___   / _ \__ ______ _')
        print('| (_ / _` | ''   \/ -_) | (_) \ V / -_) ''_|')
        print(' \___\__,_|_|_|_\___|  \___/ \_/\___|_|')


    def store_experience(self, p):
        for i, x in enumerate(p.memory.memory):
            p.replay.add(p.memory.memory[i])
    
    def load_model(self, p): 
        p.model.nn.load('saved\\' + p.name + '_model.h5')
        print('{} | Model {}_model.h5 Successfully Loaded...'.\
        format(timestamp(), p.name))

    def save_model(self, p):
        '''---Save full model to single .h5 file---'''
        p.model.nn.save(p.saves + '/' + p.name + '_model.h5', 
                        overwrite = True)
        print('{} | {} Model Saved to {}_model.h5...'.\
        format(timestamp(), p.alg, p.name))

        '''---Save model weights to separate .h5 file---'''
        p.model.nn.save_weights(p.saves +\
        '/' + p.name + '_weights.h5', overwrite = True)
        print('{} | {} Model Weights Saved in {}_weights.h5...'.\
        format(timestamp(), p.alg, p.name))

        '''---Save model structure as JSON file---'''
        with open(p.saves + '/' + p.name + '.json', 'w+') as f:
            json.dump(p.model.nn.to_json(), f)
        f.close()
        print('{} | {} Model Structure Saved in {}.json...'.\
        format(timestamp(), p.alg, p.name))


    def action(self, state, p):
        ''' Observation period occurs once, spans across episodes
            * Fills replay memory with random training data
            * After observation, training begings, epsilon anneals
            * Determines Exploration or Exploitation probability
        '''
        with p.graph.as_default(): 
            p.Qs = p.model.nn.predict(state)
        p.randQ = rand.randrange(p.A)            # explore action: idx random
        p.maxQ = np.argmax(p.Qs)                 # exploit action: idx of maxQ

        if rand.random() <= p.E or p.G <= p.observe:
            '''Explore if rand <= Epsilon or Observing'''
            p.meth = 'Explore'                 # always random if observing
            p.idx = p.randQ
        else:
            '''---Follow Greedy Policy for max Q values---'''   
            p.meth = 'Exploit'  # prob of predicting Q increases as E anneals
            p.idx  = p.maxQ


    def replay(self, p):
        '''---Select Random Batch of Experience for Training---'''
        bat = p.memory.sample(p.batch)

        '''!!!! Pick up here - the addition of Mongo Replay DB needs testing!!!!!!'''
        #replay_bat = p.replay.random_sample(p.batch)
        # parse information from sample experience
        s_bat    = np.array([e[0][0, :, :, :,] for e in bat])  # states
        a_bat    = np.array([e[1] for e in bat])               # actions
        r_bat    = np.array([e[2] for e in bat])               # rewards
        s1_bat   = np.array([e[3][0, :, :, :,] for e in bat])  # states'
        term_bat = np.array([e[4] for e in bat])               # terminal

        '''---Check for Terminal, Discount Future Rewards---'''
        with p.graph.as_default(): 
            tarQ_bat = p.model.nn.predict(s_bat)       # targetQ values
            Q_bat = p.model.nn.predict(s1_bat)         # future Q values

        '''---Logic to Update maxQ Values'''
        for i, x in enumerate(term_bat):
            '''---tarQ_bat[i] = 0 if Terminal---'''
            tarQ_bat[i, [np.argmax(a_bat[i])]] = (0) if x else \
                r_bat[i] * p.gamma * np.max(Q_bat[i])

        '''---Train using single gradient decent, loss, and Adam---
            * Fit model with training data (x) and targets (y)'''
        with p.graph.as_default():
            p.model.nn.train_on_batch(s_bat, tarQ_bat)
            # K.clear_session()
        
        '''---Decay Future Exploration Probability---'''
        if p.E > p.term_e: p.E -= (p.init_e - p.term_e) / p.anneal          


    def preprocess(self, state, p):
        '''---Preprocess frames for neural network---
            * [80, 80, 4]
            * Reorient frame from (Y, X) to (X, Y)
            * Resize image [512 x 288] -> [80 x 80] 
            * Convert to grayscale '''
        state = cv2.transpose(state)  # flips image from (Y, X) to (X, Y)
        
        p.frames.append(state)        # save image of frame
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (80, 80))
        state = cv2.threshold(state, 1, 255,
                                cv2.THRESH_BINARY)[1]
        p.trans_frames.append(state)  # save transformed frame
        return state


    def game_over(self, ep, p):
        '''---Halt play, display stats, end current episode'''
        ep +=1  # increment to correct episode number
        print('{} |   '.format(timestamp()),
                'Game: {:<7}'.format(ep),
                'Frames: {:<7}'.format(p.t), 
                'Reward: {:<8.2f}'.format(p.r_ep),
                'Score: {}'.format(p.s_ep))
        p.s_t1 = np.zeros(p.s_t.shape)          # clear next state
        p.R.append((ep, p.r_ep))                # log episode reward
        p.Sc.append((ep, p.s_ep))               # log episode score
        p.T.append((ep, p.t))                   # log episode

        '''---Save GIF of Episodes with High Enough Scores---'''
        if p.s_ep >= p.gif_score: 
            print('{} | Saving Episode {} as GIF...'.\
            format(timestamp(), ep))
            gif_path = 'learner/logs/gifs/ep{}_{}.gif'.\
            format(ep, self.sess_id)
            trans_path = 'learner/logs/gifs/trans_ep{}_{}.gif'.\
            format(ep, self.sess_id)
            imageio.mimsave(gif_path, p.frames) 
            imageio.mimsave(trans_path, p.trans_frames)
            print('{} | GIF saved successfully...\n'.\
            format(timestamp()))
        p.t = p.steps     # set as last step in episode


    def log_session(self, p, elapsed):
        '''---Log action, network, and frame information---'''
        session_log  = {
            'sess_id': self.sess_id, 'k': p.k, 'time': elapsed,
            'state_size': p.S, 'learn_rate': p.lr, 
            'memory_size': p.buffer, 'batch_size': p.batch, 
            'episodes': p.episodes, 'max_ep_steps': p.steps,
            'score_target': p.target, 'gif_score': p.gif_score, 
            'observe': p.observe, 'anneal': p.anneal,
            'action_size': p.A, 'filter_size': p.H
        }

        with open(p.logs + 'sessions.csv', 'a+', 
                  newline='') as f: 
            writer = csv.DictWriter(f, session_log.keys())
            if os.stat(self.s_log).st_size == 0:
                writer.writeheader()      # add header row to new file
            writer.writerow(session_log)
        f.close()


    def log_state(self, ep, p):
        '''---Log action, network, and frame information---'''
        ep +=1  # increment to correct episode number
        game_log  = {
            'ep_id': p.ep_id, 'ep': ep, 'step': p.t, 
            'g_step': p.G, 'E': p.E,'target': p.target, 'score': p.s_ep,
            'a_t': p.a_t, 'time': timestamp(),'term': str(p.terminal), 
            'meth': p.meth, 'msg': p.msg, 'Qs': p.Qs[0], 
            'maxQ': p.maxQ, 'randQ': p.randQ, 'r_t': p.r_t, 
            'player_y': p.game.playery, 'pipe1': p.game.pipe1, 
            'pipe2': p.game.pipe2, 'upper': p.game.upperPipes,
            'lower': p.game.lowerPipes, 'gap_pos': p.game.gapPos, 
            'vel_y': p.game.playerVelY,
            'flap': str(p.game.playerFlapped),
            'acc_flap': p.game.playerFlapAcc,
        }

        with open(self.s_log, 'a+', newline='') as f: 
            writer = csv.DictWriter(f, game_log.keys())
            if os.stat(self.s_log).st_size == 0:
                writer.writeheader()      # add header row to new file
            writer.writerow(game_log)
        f.close()


def timestamp(): return time.strftime('%Y-%m-%d@%H:%M:%S')