import sys
sys.path.append('game/')
sys.path.append('learner/')

import os
import cv2
import tensorflow as tf
import random as rand  
import numpy as np
import game.flappy as flappy
import matplotlib.pyplot as plt
import matplotlib.image as img

from learner.deep_q import DeepQ
from learner.experience_replay import Buffer
from collections import deque

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
# ---Global Constant Parameters------------------------------------------------
'''---Game and Action-State Settings---'''
NAME = 'flappy'         # identifier for log files
MODE = 'hard'           # sets game mode to hard
FPS  = 30               # frames per second in emulation
S    = 4                # number of states
A    = 2                # number of possible actions
H    = 64               # number of hidden layers
K    = 1                # frames per action

'''---Observation and Exploration---'''
# Parameter values have been slashed for testing

EPISODES     = 10          # number times to play game                         10000
STEPS        = 200         # max steps per episode                        FPS * 3600
OBSERVE      = STEPS       # observation steps pre training                STEPS * 5
EXPLORE      = STEPS * 3   # steps to anneal epsilon                      STEPS * 30
INIT_EPSILON = 0.3         # initial value of epsilon
TERM_EPSILON = 0.001       # terminal value of epsilon

'''---Algorithm Parameters---'''
TARGET       = 40          # goal; sets the bar for success
GAMMA        = 0.99        # discount factor of future rewards
LR           = 0.01        # learning rate

'''---Replay Memory---'''
BUF_SIZE  = OBSERVE        # number of steps to track
BATCH     = 32             # minibatch size                                       64

'''---Log Files---'''
A_LOG  = open('learner/logs/action.log', 'w')   # actions log
S_LOG  = open('learner/logs/state.log', 'w')    # state log
H_LOG  = open('learner/logs/hidden.log', 'w')   # hidden log
# -----------------------------------------------------------------------------

'''Your mission, should you choose to accept it...'''
class Agent:
    def __init__(self):
        super(Agent, self).__init__()

        '''---Initialize Session---'''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config = config) 

        '''---Hook in DeepQ Neural Network---'''
        self.RL = DeepQ(state_size  = S,
                        action_size = A,   
                        hidden_size = H)

        '''---Initialize neural network---'''    
        self.s, self.out, self.layer = self.RL.network()

        '''---Persistent tracking---'''
        self.buffer  = Buffer(BUF_SIZE, BATCH)          # replay memory
        self.R       = []            # tracks total episode rewards
        self.Sc      = []            # tracks total episode scores
        self.T       = []            # tracks total steps in episode

        self.step    = 0             # tracks steps across episodes
        #self.im      = plt.figure()  # prepare frame for rendering

    def train(self, target = TARGET, mode = MODE):
        '''---Begin game emulation---'''
        for ep in range(1, EPISODES):
            '''---START EPISODE---'''
            #print('---EPISODE {}--------------------------------------------'.\
            #      format(ep))
            self.game = flappy.GameState(target, mode)  # game emulation
            E    = INIT_EPSILON                         # training epsilon
            r_ep = 0                                    # episode reward
            t    = 0                                    # transition number

            '''---Initialize first state with no action---'''
            a = tf.placeholder(tf.float32, [None, A], name = 'actions')
            wake = np.zeros(A)
            wake[0] = 1
            # color_img, reward, score, terminal -> empty
            x_t_c, r_0, s_ep, terminal = self.game.step(wake)
    
            '''Preprocess image data, and stack into state_size layers'''
            x_t = self.preprocess(x_t_c, t)             # gray_img', state'
            self.s_t = np.stack((x_t, x_t, x_t, x_t), 
                                 axis = 2)              # stack 4 frames

            '''---Prepare neural network to run emulation---'''
            y, self.loss = self.loss_fn(a)    # targetQ and Loss function
            play = tf.train.AdamOptimizer(LR).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

            '''---OBSERVATION LOOP--------------------------------------------------'''
            while t < STEPS:  
                t += 1                                   # limit episode max steps
                self.step += 1                           # increment total steps
                Qs = self.greedy([self.s_t])               # greedy maxQ policy
                maxQ = np.argmax(Qs)            # exploit action
                randQ = rand.randrange(A)       # explore action
                '''---Check if observing or training---
                * Observation period occurs once, spans across episodes
                * Fills replay memory with random training data
                * Training begins after observations, epsilon begins annealing
                  * Determines Exploration or Exploitation probability'''
                a_t = np.zeros([A])
                
                if rand.random() <= E or self.step <= OBSERVE:
                    '''Explore if rand <= Epsilon or in observation period'''
                    meth = 'Explore'              # always random if observing
                    idx = randQ
                else:
                    '''Exploit knowledge from previous observations'''
                    meth = 'Exploit'
                    idx  = maxQ
                a_t[idx] = 1                      # set action
                flap = False if a_t[0] else True 
                #print('Step: {}  |  {}  |  Flap: {}'.\
                #      format(t, meth, flap))
                '''---Send action to emulator---'''
                x_t1_c, r_t, s_ep, terminal = self.game.step(a_t)
                x_t1 = self.preprocess(x_t1_c, t)
                x_t1 = np.reshape(x_t1, (80, 80, 1))  # add channels dimension
                self.s_t1 = np.append(self.s_t[:, :, :3], x_t1, axis=2)

                r_ep += r_t                           # update reward total
                '''---Store experience in memory---'''
                # self.memory.maxlen = self.BUF_SIZE
                # (state, action, reward, state', terminal)
                self.add_exp(a_t, r_t, terminal)
                self.saver  = tf.train.Saver(max_to_keep = 25)  # checkpoint
                self.load_save(False)                           # restore 
    
                '''---TRAINING LOOP---------------------------------------------'''
                if self.step > OBSERVE:       # begin training
                    E = self.decay(E)         # decay exploration probability
                    '''---Select random batch of sample for training---'''
                    mini = self.buffer.sample(BATCH)

                    # parse sample information
                    s_j_bat  = np.array([d[0] for d in mini])    # states
                    a_bat    = np.array([d[1] for d in mini])    # actions
                    r_bat    = np.array([d[2] for d in mini])   # rewards
                    s_j1_bat = np.array([d[3] for d in mini])   # states'

                    '''---Return game state information for each sample---'''
                    y_bat    = []        # empty maxQ matrix
                    Q_bat  = self.greedy(s_j1_bat)

                    '''---Calculate maxQ for each state---'''
                    for i in range(0, len(mini)):
                        terminal = mini[i][4]          # check for game over   
                        if terminal:                   # if game over
                            y_bat.append(r_bat[i])     # log reward 

                        else:                          # discount maxQ
                            discount = GAMMA * np.max(Q_bat[i])
                            r = r_bat[i] + discount    # add to reward
                            y_bat.append(r)            # log reward

                    '''---Train using gradient decent, loss and Adam---'''
                    play.run(feed_dict = {
                                y : y_bat,
                                a : a_bat,
                                self.s : s_j_bat }) 
                #if terminal: plt.show()
                '''---Check for game over---'''
                if terminal: t = self.game_over(t, ep, r_ep, s_ep)
                    # log episode stats and flag episode as finished

                '''---Save and log progress every 10000 steps---'''
                # Parameter values have been slashed for testing
                if self.step % 1000 == 0:                                   # self.step % 10000
                    self.new_save(ep, t) # new checkpoint
                    '''---Log state information---'''
                    if t <= OBSERVE: state = 'observe'  # get state type
                    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                        state = 'explore'
                    else: state = 'train'
                    S_LOG.write('EPISODE', ep,
                                    '/ STEP', t, 
                                    '/ STATE', state, 
                                    '/ METHOD', meth,
                                    '/ EPSILON', E,
                                    '/ ACTION', idx,
                                    '/ REWARD', r_t,
                                    '/ MAX_Q %e' % np.max(Qs) + '\n')

                    '''---Log action, network, and frame information---'''
                    A_LOG.write(','.join([str(x) for x in Qs]) + '\n')
                    H_LOG.write(','.join([str(x) for x in self.layer.eval(\
                                     feed_dict = {self.s : [self.s_t]})[0]]) + '\n')
                    # Save frame as image
                    cv2.imwrite('logs/images/' + NAME + '_ep' + str(ep)\
                                + '_frame' + str(t) + '.png', x_t_c)
                '''---Progress to next step/transition---'''
                self.s_t = self.s_t1                    

    
    def greedy(self, s_t):
        '''---Follow greedy policy to determine maxQ---'''
        feed = {self.s: s_t}
        return self.out.eval(feed_dict = feed)

    def resample(self, x_t):
        ''' * Reorient frame from (Y, X) to (X, Y)
            * Resize image [432 x 288] -> [80 x 80] 
            * Convert to grayscale '''
        x_t = cv2.transpose(x_t)  # flips image from (Y, X) to (X, Y)
        x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)
        x_t = cv2.resize(x_t, (80, 80))
        return x_t

    def threshold(self, x_t):
        '''---Normalize image from [0, 255] -> [0, 1] tones---
        * input must be greyscale image
          * separates black from all other tones
        * tone value > 1  = white
        * tone value <= 1 = black '''
        x_t = cv2.threshold(x_t,
                            1, 255,
                            cv2.THRESH_BINARY)[1]
        #plt.imshow(x_t, cmap = 'gray', vmin = 0, vmax = 255)
        return x_t

    def preprocess(self, x_t, t):
        '''---Preprocess frames for neural network---
        * state is passed as a stack of four sequential frames
        * first t of episode creates stack of state_size frames
        * subsequent t removes oldest frame, appends image to stack
          * [80, 80, 4]'''
        x_t = self.resample(x_t)
        x_t = self.threshold(x_t)
        return x_t

    def run_emulation(self):
        '''---Initialize RL Agent to train on emulation using neural network'''
        self.train()
        
    def add_exp(self, a_t, r_t, term):
        '''---Save experience to replay buffer---'''
        self.buffer.add(self.s_t, a_t, r_t, self.s_t1, term)

    def decay(self, E):
        '''---Anneal the value of E over exploration period---'''
        # stops annealing when E = self.TERM_EPSILON
        if E > TERM_EPSILON:
            E -= (INIT_EPSILON - TERM_EPSILON) / EXPLORE
        return E

    def loss_fn(self, actions):
        '''---Defines the loss function of action state---'''
        # Q has 2 dimensions, each corresponding to an action
        # Set training target to max Q
        #a_ot    = tf.one_hot(actions, A)
        targetQ = tf.placeholder(tf.float32, [None], name = 'target')
        Q = tf.reduce_sum(tf.multiply(self.out, actions), 
                          axis = 1)
        loss = tf.reduce_mean(tf.square(targetQ - Q))
        return targetQ, loss

    def new_save(self, ep, t):
        '''---Save checkpoint as new file---'''
        self.saver.save(self.sess, 
                       'saved/' + NAME + '_ep' + str(ep)+ '_frame' + str(t),
                        global_step=self.step)

    def load_save(self, restore = False):
        '''---Restore data from previous checkpoint---'''
        checkpoint = tf.train.get_checkpoint_state('saved/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('--Loaded Checkpoint--\n' + \
                  checkpoint.model_checkpoint_path + '\n')
        else: return

    def game_over(self, t, ep, r_ep, s_ep):
        '''---Halts further steps, logs stats, and ends the current episode'''
        # Execute if statement, skip remaining while statement
        self.s_t1 = np.zeros(self.s_t.shape)          # clear next state
        self.R.append((ep, r_ep))           # log episode reward
        self.Sc.append((ep, s_ep))          # log episode score
        self.T.append((ep, t))              # log episode steps
        # Display Episode Stats
        #print('  ___                   ___')         
        #print(' / __|__ _ _ __  ___   / _ \__ _____ _ _')
        #print('| (_ / _` | ''   \/ -_) | (_) \ V / -_) ''_|')
        #print(' \___\__,_|_|_|_\___|  \___/ \_/\___|_|')
        print('\n\nEpisode: {}'.format(ep),
              '| Score: {}  '.format(s_ep),
              '| Steps: {}  '.format(t), 
              '| Episode Reward: {:.3f}  '.format(r_ep),
              '\n' + '-' * 56)
        t = STEPS               # set as last step in episode
        return t