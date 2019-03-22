import sys
sys.path.append('game/')
sys.path.append('learner/')

import os
import cv2
import csv
import time
import uuid
import imageio
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

EPISODES     = 1000           # number times to play game              
STEPS        = FPS * 60 * 5   # max steps per episode  (10 minutes)
OBSERVE      = STEPS // 2     # observation steps pre training
EXPLORE      = STEPS * 5      # steps to anneal epsilon 
INIT_EPSILON = 0.3            # initial value of epsilon
TERM_EPSILON = 0.001          # terminal value of epsilon

'''---Algorithm Parameters---'''
TARGET       = 40          # goal; sets the bar for success
GAMMA        = 0.99        # discount factor of future rewards
LR           = 0.01        # learning rate

'''---Replay Memory---'''
BUF_SIZE  = OBSERVE        # number of steps to track
BATCH     = 64             # minibatch size 
# -----------------------------------------------------------------------------

'''Your mission, should you choose to accept it...'''
class Agent:
    def __init__(self):
        super(Agent, self).__init__()

        '''---Initialize Session---'''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config = config) 
        self.sess_id = time.strftime('%Y%m%d-%H%M%S')

        '''---Hook in DeepQ Neural Network---'''
        self.RL = DeepQ(state_size  = S,
                        action_size = A,   
                        hidden_size = H)

        '''---Persistent tracking---'''
        self.buffer  = Buffer(BUF_SIZE, BATCH)          # replay memory
        self.R       = []            # tracks total episode rewards
        self.Sc      = []            # tracks total episode scores
        self.T       = []            # tracks total steps in episode
        self.step    = 0             # tracks steps across episodes

        self.saver = tf.train.Saver(max_to_keep = 5)    # checkpoint
        self.load_save(True)                            # restore 

        '''---Log Files---'''
        self.s_log = 'learner/logs/session_{}.csv'.format(self.sess_id)   # session log

    def train(self, target = TARGET, mode = MODE):
        '''---Initialize neural network---'''    
        self.s, self.out, self.layer = self.RL.network()

        a = tf.placeholder(tf.float32, [None, A], name = 'actions')
        y, loss = self.loss_fn(a)    # targetQ and Loss function
        train = tf.train.AdamOptimizer(LR).minimize(loss)

        print('\nNeural Network Initialized...\n')
        print('Begining observation period...\n')
        training = False

        '''---Begin game emulation---'''
        for ep in range(1, EPISODES + 1):
            '''---START EPISODE---'''
            #print('---EPISODE {}--------------------------------------------')
            
            self.ep_id = uuid.uuid1()                        # unique identifier for episode
            self.game = flappy.GameState(target, mode)  # game emulation
            E    = INIT_EPSILON                         # training epsilon
            r_ep = 0                                    # episode reward
            t    = 0                                    # transition number
            frames = []
            trans_frames = []

            game_log  = {'sess_id' : self.sess_id, 'ep_id' : self.ep_id, 
                         'ep' : ep, 'frame' : None, 'score' : None,
                         's_t' : [], 'a_t' : [], 'r_t' : None,
                         's_t1' : [], 'term' : None, 'meth' : None, 
                         'msg' : None, 'E' : None, 'Qs' : [], 'maxQ' : None, 
                         'img' : [], 'layer' : self.layer, 'out' : self.out}

            '''---Initialize first state with no action---'''
            wake = np.zeros(A)
            wake[0] = 1
            # color_img, reward, score, terminal -> empty
            x_t_c, r_0, r_ep, s_ep, terminal, msg = self.game.step(wake)

            '''Preprocess image data, and stack into state_size layers'''
            x_t = self.preprocess(x_t_c, t, frames, trans_frames)    # gray_img', state'
            self.s_t = np.stack((x_t, x_t, x_t, x_t), 
                                 axis = 2)                           # stack 4 frames

            self.sess.run(tf.global_variables_initializer())

            '''---OBSERVATION LOOP--------------------------------------------------'''
            while t < STEPS:  
                t += 1                                   # limit episode max steps
                self.step += 1                           # increment total steps
                Qs = self.greedy([self.s_t])             # greedy maxQ policy
                maxQ = np.argmax(Qs)                     # exploit action
                randQ = rand.randrange(A)                # explore action
                '''---Check if observing or training---
                * Observation period occurs once, spans across episodes
                * Fills replay memory with random training data
                * Training begins after observations, epsilon begins annealing
                  * Determines Exploration or Exploitation probability'''

                a_t = np.zeros([A])
                
                if t % K == 0:
                    if rand.random() <= E or self.step <= OBSERVE:
                        '''Explore if rand <= Epsilon or in observation period'''
                        meth = 'Explore'                 # always random if observing
                        idx = randQ
                    else:
                        '''Exploit knowledge from previous observations'''
                        meth = 'Exploit'
                        idx  = maxQ
                else: 
                    meth = 'Wait'
                    idx = 0                            # don't flap
                a_t[idx] = 1                             # set action
                flap = False if a_t[0] else True 
                
                '''---Send action to emulator---'''
                x_t1_c, r_t, r_ep, s_ep, terminal, msg = self.game.step(a_t)
                x_t1 = self.preprocess(x_t1_c, t, frames, trans_frames)
                x_t1 = np.reshape(x_t1, (80, 80, 1))  # add channels dimension
                self.s_t1 = np.append(self.s_t[:, :, :3], x_t1, axis=2)
                r_ep += r_t                           # update reward total

                '''---Store experience in memory---'''
                # self.memory.maxlen = self.BUF_SIZE
                # (state, action, reward, state', terminal)
                self.add_exp(a_t, r_t, terminal)

                #print('Step: {}  |  {}  |  Flap: {}  |  Reward: {:.3f}  |  {}'.\
                #      format(t, meth, flap, r_t, msg))
                '''---Check for game over---'''
                if terminal: 
                    # Display Episode Stats
                    print('Episode: {:<6}\t'.format(ep),
                          'Steps: {:<6}\t'.format(t), 
                          'Reward: {:<12.3f}\t'.format(r_ep),
                          'Score: {}'.format(s_ep),
                        )
                    t = self.game_over(t, ep, r_ep, s_ep, frames, trans_frames)
                    continue
                # log episode stats and flag episode as finished
    
                '''---TRAINING LOOP---------------------------------------------'''
                if self.step > OBSERVE:       # begin training
                    if training == False:
                        training = True
                        print('\nObservation complete...')
                        print('Beginning Training Period...')
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
                    train.run(feed_dict = {
                                y : y_bat,
                                a : a_bat,
                                self.s : s_j_bat })

                '''---Log action, network, and frame information---'''
                game_log['frame'] = t
                game_log['score'] = s_ep
                game_log['s_t'] = str(self.s_t).split('\n')
                game_log['a_t'] = a_t
                game_log['r_t'] = r_t
                game_log['s_t1'] = str(self.s_t1).split('\n')
                game_log['term'] = str(terminal)
                game_log['meth'] = meth
                game_log['msg'] = msg
                game_log['E'] = E
                for q in Qs: 
                    game_log['Qs'].append(str(q))
                game_log['maxQ'] = maxQ
                game_log['img'] = str(x_t_c).split('\n')

                with open(self.s_log, 'a+', newline='') as f: 
                    writer = csv.DictWriter(f, game_log.keys())
                    if os.stat(self.s_log).st_size == 0:
                        writer.writeheader()         # add header row to new file
                    writer.writerow(game_log)
                f.close()

                '''---Save progress every (steps % n) frames---'''
                if self.step % 3000 == 0 or \
                   self.step == OBSERVE + 1 or \
                   (ep == EPISODES and terminal):
                    self.new_save(ep)                # new checkpoint
                    continue

                # Save frame as image
                #cv2.imwrite('learner/logs/images/' + NAME + '_ep' + str(ep)\
                #            + '_frame' + str(t) + '.png', 
                #            cv2.transpose(\
                #                cv2.cvtColor(x_t_c, cv2.COLOR_BGR2RGB)))
                
                '''---Progress to next step/transition---'''
                self.s_t = self.s_t1       

        # Exit PyGame at the end of training                     
        self.game.quit_game()
        print('\nTraining Session Complete!')
        print('  ___                   ___')         
        print(' / __|__ _ _ __  ___   / _ \__ ______ _')
        print('| (_ / _` | ''   \/ -_) | (_) \ V / -_) ''_|')
        print(' \___\__,_|_|_|_\___|  \___/ \_/\___|_|')
    
    def greedy(self, s_t):
        '''---Follow greedy policy to determine maxQ---'''
        feed = {self.s: s_t}
        return self.out.eval(feed_dict = feed)

    def preprocess(self, x_t, t, frames, trans_frames):
        '''---Preprocess frames for neural network---
        * state is passed as a stack of four sequential frames
        * first t of episode creates stack of state_size frames
        * subsequent t removes oldest frame, appends image to stack
          * [80, 80, 4]
        * Reorient frame from (Y, X) to (X, Y)
        * Resize image [432 x 288] -> [80 x 80] 
        * Convert to grayscale '''
        x_t = cv2.transpose(x_t)  # flips image from (Y, X) to (X, Y)
        #rgb = cv2.cvtColor(x_t, cv2.COLOR_BGR2RGB)
        
        frames.append(x_t)  # save pygame frames
        x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)
        x_t = cv2.resize(x_t, (80, 80))
        x_t = cv2.threshold(x_t,
                            1, 255,
                            cv2.THRESH_BINARY)[1]
        trans_frames.append(x_t)  # save pygame transformed frames
        #plt.imshow(x_t, cmap = 'gray', vmin = 0, vmax = 255)
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

    def new_save(self, ep):
        '''---Save checkpoint as new file---'''
        self.saver.save(self.sess, 
                       'saved/' + NAME + '-ep{}'.format(str(ep)),
                        global_step=self.step)
        print('\n--Created New Checkpoint--\n')

    def load_save(self, restore = False):
        '''---Restore data from previous checkpoint---'''
        try:
            ckpt = tf.train.get_checkpoint_state('saved/')
            if restore and ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            else: print('--No Restore Attempt Made--')
        except: print('--No Checkpoints to Restore--')

    def game_over(self, t, ep, r_ep, s_ep, frames, trans_frames):
        '''---Halts further steps, logs stats, and ends the current episode'''
        # Execute if statement, skip remaining while statement
        self.s_t1 = np.zeros(self.s_t.shape)          # clear next state
        self.R.append((ep, r_ep))           # log episode reward
        self.Sc.append((ep, s_ep))          # log episode score
        self.T.append((ep, t))              # log episode
        # Save videos of episode as GIFs if past first hit spot
        if t > 49: 
            print('Saving Episode {} as GIF...'.format(ep))
            gif_path = 'learner/logs/gifs/ep{}_{}.gif'.format(ep, self.sess_id)
            trans_path = 'learner/logs/gifs/trans_ep{}_{}.gif'.\
                format(ep, self.sess_id)
            imageio.mimsave(gif_path, frames)   # save gif of episode (normal)
            imageio.mimsave(trans_path, trans_frames)   # save gif of episode (transformed)
            print('GIF Saved Successfully...')
        t = STEPS               # set as last step in episode
        return t