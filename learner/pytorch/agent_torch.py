import sys
sys.path.append('game/')
sys.path.append('learner/')

import os
import cv2

import torch 
import torch.transforms as TN
import torch.transforms.functional as TF
import torch.nn.functional as NN
import torch.nn as nn
import torch.utils.data as data
import torchvision

import random as rand  
import numpy as np
import game.flappy as flappy
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animate

from learner.deep_q_torch import DeepQ
from learner.experience_replay import Buffer
from collections import deque

'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    agent.py
Author:  Whitney King
Date:    March 8, 2019

References:
    PyTorch DeepQ CNN Tutorial
    Author: Morvan Zhou
    https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/\
        tutorial-contents/401_CNN.py

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

EPISODES     = 10          # number times to play game (epoch)                  1000
STEPS        = 500         # max steps per episode                         FPS * 900
OBSERVE      = STEPS       # observation steps pre training                STEPS * 5
EXPLORE      = STEPS * 3   # steps to anneal epsilon                      STEPS * 30
INIT_EPSILON = 0.5         # initial value of epsilon
TERM_EPSILON = 0.001       # terminal value of epsilon

'''---Replay Memory---'''
BUF_SIZE  = OBSERVE        # number of steps to track
BATCH     = 32             # minibatch size                                       64

'''---Algorithm Parameters---'''
TARGET       = 40          # goal; sets the bar for success
GAMMA        = 0.99        # discount factor of future rewards
LR           = 0.01        # learning rate

'''---Log Files---'''
A_LOG  = open('learner/logs/action.log', 'w')   # actions log
S_LOG  = open('learner/logs/state.log', 'w')    # state log
H_LOG  = open('learner/logs/hidden.log', 'w')   # hidden log
# -----------------------------------------------------------------------------

'''Your mission, should you choose to accept it...'''
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        '''---Initialize Session---'''
        self.sess   = torch.Session()           # initialize session

        '''---Hook in DeepQ Neural Network---'''
        self.RL      = DeepQ(state_size  = S,
                             action_size = A,   
                             hidden_size = H)

        '''---Persistent tracking---'''
        self.buffer  = Buffer(BUF_SIZE, BATCH)          # replay memory
        self.R       = []            # tracks total episode rewards
        self.Sc      = []            # tracks total episode scores
        self.T       = []            # tracks total steps in episode
        self.step    = 0             # tracks steps across episodes
        self.im      = plt.figure()  # prepare frame for rendering

    def train(self, target = TARGET, mode = MODE):
        '''---Initialize neural network---'''  
        # state, network_output, fully-connected_layer      
        s, out, fc1_h = self.RL.network()

        '''---Begin game emulation---'''
        for ep in range(1, EPISODES):
            '''---START EPISODE---'''
            self.game = flappy.GameState(target, mode)  # game emulation
            E    = INIT_EPSILON                         # training epsilon
            r_ep = 0                                    # episode reward
            t    = 0                                    # transition number
            '''---Initialize first state with no action---'''
            a = a_t = wake = np.zeros([A], dtype = 'int32')
            wake[0] = 1                                 # no action
            # color_img, reward, score, terminal
            x_t_c, r_0, s_ep, terminal = self.game.step(wake) # empty first step

            '''Preprocess image data, and stack into state_size layers'''
            s_t = self.preprocess(x_t_c, t)        # gray_img', state'

            '''---Prepare neural network to run emulation---'''
            y, self.loss = self.loss_fn(a, out)    # targetQ and Loss function
            play = torch.optim.Adam(LR).minimize(self.loss)

            '''---Start selecting actions and stepping through states---'''
            while t < STEPS:                        # limit episode max steps
                self.step += 1                      # increment total steps
                Qs  = self.greedy(out, s, s_t)[0]   # greedy maxQ policy
                idx = 0                             # action index
                '''---Check if observing or training---
                * Observation period occurs once, spans across episodes
                * Fills replay memory with random training data
                * Training begins after observations, epsilon begins annealing
                  * Determines Exploration or Exploitation probability'''

                if rand.random() <= E or self.step <= OBSERVE:
                    '''Explore if rand <= Epsilon or in observation period'''
                    meth = 'explore'              # always random if observing
                    idx = rand.randrange(A)       # random action
                else:
                    '''Exploit knowledge from previous observations'''
                    meth = 'exploit'
                    idx  = np.argmax(Qs)          # best action
                a_t[idx] = 1                      # flag action
                '''---Send action to emulator---'''
                x_t1_c, r_t, s_ep, terminal = self.game.step(a_t)
                s_t1 = self.preprocess(x_t1_c, t)
                
                r_ep += r_t                       # update reward total
                '''---Store experience in memory---'''
                # self.memory.maxlen = self.BUF_SIZE
                # (state, action, reward, state', terminal)
                self.add_exp(s_t, a_t, r_t, s_t1, terminal)
                self.saver  = tf.train.Saver(max_to_keep = 25)  # checkpoint
                self.sess.run(tf.global_variables_initializer())

                self.load_save(False)                           # restore 
                '''---Check for game over---'''
                if terminal:
                    '''---GAME OVER: END EPISODE---'''
                    # log episode stats and flag episode as finished
                    t = self.game_over(t, s_t, s_t1, ep, r_ep, s_ep, E)
                    continue                  # skip to next episode
                '''---Check for training period---'''
                if self.step > OBSERVE:       # begin training
                    E = self.decay(E)         # decay exploration probability
                    '''---Select random batch of sample for training---'''
                    mini = self.buffer.sample(BATCH)
                    # parse sample information
                    s_j_bat  = [d[0] for d in mini]    # states
                    a_bat    = [d[1] for d in mini]    # actions
                    r_bat    = [d[2] for d in mini]    # rewards
                    s_j1_bat = [d[3] for d in mini]    # states'
                    '''---Return game state information for each sample---'''
                    y_bat    = []        # empty maxQ matrix
                    Q_bat  = self.greedy(out, s, s_j1_bat)
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
                             s : s_j_bat }) 
                '''---Render emulation on screen---'''
                # Update rendering @ 10 FPS
                self.animate = animate.FuncAnimation(self.im, 
                                                     self.get_image(x_t1_c),
                                                     interval = 100,
                                                     blit = True)
                plt.show()
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
                    H_LOG.write(','.join([str(x) for x in fc1_h.eval(\
                                     feed_dict = {s : [s_t]})[0]]) + '\n')
                    # Save frame as image
                    cv2.imwrite('logs/images/' + NAME + '_ep' + str(ep)\
                                + '_frame' + str(t) + '.png', x_t_c)
                '''---Progress to next step/transition---'''
                s_t = s_t1
                t  += 1

    
    def greedy(self, out, s, s_t):
        '''---Follow greedy policy to determine maxQ---'''
        #s_t = s_t.reshape((1, *s_t.shape))
        feed = {s: [s_t]}
        print('feed: ', feed)
        return out.eval(feed_dict = feed)

    def resample(self, x_t):
        ''' * Reorient frame from (Y, X) to (X, Y)
            * Resize image [512 x 288] -> [80 x 80] 
            * Convert to grayscale '''
        x_t = TN.Compose(
            TF.to_grayscale(x_t),
            TF.resize(x_t, [80, 80]),
            TF.to_tensor(x_t)
        )
        # plt.imshow(x_t, cmap = 'gray', vmin = 0, vmax = 255)
        return x_t

    def threshold(self, x_t):
        '''---Normalize image from [0, 255] -> [0, 1] tones---
          * separates black from all other colors
        * tone value >= 1  = white
        * tone value < 1 = black '''
        x_t = nn.Threshold(1, 1)  # if color > 1, flip to 1

        # plt.imshow(x_t, cmap = 'gray', vmin = 0, vmax = 255)
        return x_t

    def preprocess(self, x_t, t):
        '''---Preprocess frames for neural network---
        * state is passed as a stack of four sequential frames
        * first t of episode creates stack of state_size frames
        * subsequent t removes oldest frame, appends image to stack
          * [80, 80, 4]'''
        x_t = self.resample(x_t)
        x_t = self.threshold(x_t)
        if t == 0: s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2) # placeholder
        else:                           # append frame to end of existing stack
            x_t = np.reshape(x_t, (80, 80, 1))
            s_t = np.append(x_t, s_t[:, :, :3], axis = 2)
        return s_t

    def get_image(self, x_t):
        '''---Gets frame and formats it as an image to render on screen---'''
        self.im = cv2.cvtColor(img.imread(x_t),cv2.COLOR_BGR2RGB)
        self.im = plt.imshow(self.im)     # prepare frame display

    def run_emulation(self):
        '''---Initialize RL Agent to train on emulation using neural network'''
        self.train()

    def game_over(self, t, s_t, s_t1, ep, r_ep, s_ep, E):
        '''---Halts further steps, logs stats, and ends the current episode'''
        # Execute if statement, skip remaining while statement
        s_t1 = np.zeros(s_t.shape)          # clear next state
        self.buffer.memory[-1][3] = s_t1    # update replay memory
        self.R.append((ep, r_ep))           # log episode reward
        self.Sc.append((ep, s_ep))          # log episode score
        self.T.append((ep, t))              # log episode steps
        # Display Episode Stats
        print('Episode:\t{}\t'.format(ep), 
              '| Score:\t{}\t'.format(s_ep),
              '| Steps:\t{}\t'.format(t), 
              '| Reward:\t{}\t'.format(r_ep),
              '| Loss:\t{}\t'.format(self.loss), 
              '| Terminal E:\t{}'.format(E),
              '-' * 80 + '\n')
        t = STEPS               # set as last step in episode
        return t
        
    def add_exp(self, s_t, a_t, r_t, s_t1, term):
        '''---Save experience to replay buffer---'''
        self.buffer.add(s_t, a_t, r_t, s_t1, term)

    def decay(self, E):
        '''---Anneal the value of E over exploration period---'''
        # stops annealing when E = self.TERM_EPSILON
        if E > TERM_EPSILON:
            E -= (INIT_EPSILON - TERM_EPSILON) / EXPLORE
        return E

    

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
        else: print ('--Created New Checkpointer--')

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        def forward(self, a, out):
            '''---Defines the loss function of action state---'''
            # Q has 2 dimensions, each corresponding to an action
            # Set training target to max Q
            a_ot    = tf.one_hot(a, self.A)
            targetQ = tf.placeholder(torch.float32, [None], name = 'target')
            Q    = tf.reduce_sum(torch.prod(out, a_ot), axis = 1)
            loss = tf.reduce_mean(tf.square(targetQ - Q))
            return targetQ, loss