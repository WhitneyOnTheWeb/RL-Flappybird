'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    agent.py
Author:  Whitney King
Date:    March 8, 2019

References:
    Udacity ML Engineer Quadcopter Project
    github.com/WhitneyOnTheWeb/deep-learning-master/blob/master/Quadcopter/tasks/takeoff.py

    Udacity ML Engineer Nanodegree Classroom
    tinyurl.com/yd7rye3w
'''
import os
import cv2
import sys
import opencv
import pprint as pp
import tensorflow as tf
import random as rand  
import numpy as np
from .deep_q import DeepQ
from .experience_replay import Buffer
from collections import deque
sys.path.append('../game/')
import game.flappy as flappy

'''Your mission, should you choose to accept it...'''
class Agent:
    # Defines the goal and provides feedback to the agent
    def __init__(self):
        # Initialize object to execute play Flappy Bird task
        self.NAME = 'flappy'  # identifier for log files
        self.FPS  = 30        # frames per second in emulation
        self.S    = 4         # number of states
        self.A    = 2         # number of possible actions
        self.H    = 64        # number of hidden layers
        self.K    = 1         # frames per action

        # Observation and Exploration Parameters
        self.EPISODES = 10000             # number times to play game
        self.STEPS    = self.FPS * 3600   # max steps per episode
        self.OBSERVE  = self.STEPS / 2    # observation steps pre training
        self.EXPLORE  = self.STEPS * 30   # steps to anneal epsilon
        self.INIT_EPSILON = 0.5           # initial value of epsilon
        self.TERM_EPSILON = 0.001         # terminal value of epsilon

        # Replay Memory
        self.BUF_SIZE  = 50000            # number of steps to track
        self.BATCH     = 32               # minibatch size
        self.memory    = Buffer(self.BUF_SIZE,
                                self.BATCH)

        # Algorithm Parameters
        self.TARGET_SCORE = 40        # goal; sets the bar for success
        self.GAMMA        = 0.99      # discount factor of future rewards
        self.TAU          = 0.001     # soft target update
        self.LR           = 0.01      # learning rate

        # Log Files
        self.A_LOG  = open('logs/readout.log', 'w')  # actions log
        self.S_LOG  = open('logs/state.log', 'w')    # state log
        self.H_LOG  = open('logs/hidden.log', 'w')   # hidden log

        # Persistent Tracking
        self.reward_list = []        # tracks total episode rewards
        self.score_list  = []        # tracks total episode scores
        self.step = 0                # count steps across all episodes

    def train(self, s, out, fc1_h, sess):
        '''----------------------Begin game emulation-----------------------'''
        sess.run(tf.initialize_all_variables())    # initialize session
        save = self.save_load(sess, False)         # load saved networks
        
        for ep in range(1, self.EPISODES):
            '''-------------------Initialize new episode--------------------'''
            game = flappy.GameState()     # start / reset game emulation
            E    = self.INIT_EPSILON      # begin observation period
            r_ep = 0                      # total episode reward
            t    = 0                      # transition number
            while t < self.STEPS:
                '''--------------Initialize empty first state---------------'''
                a = tf.placeholder("float", [None, self.A])
                self.step += 1                       # count total steps
                Q, targetQ, loss = loss(out, a)      # Qs, Loss, and Optimizer
                play = tf.train.AdamOptimizer(self.LR).minimize(loss)
                a_t, wake = np.zeros(self.A)              # empty action matrix
                wake[0] = 1                                # no action
                x_t, r_0, s_ep, terminal = game.step(wake) # first step
                x_t, s_t = self.preprocess(x_t)           # Preprocess data

                '''----------------Start selecting actions------------------'''
                out_t = self.greedy(out, s, s_t)[0]    # follow greedy policy
                idx   = 0                              # action index
                if rand.random() <= E or t <= self.OBSERVE:
                    '''Explore if rand <= Epsilon or in observation period'''
                    meth = 'explore'
                    idx = rand.randrange(self.A)  # random action
                else:
                    ''' Exploit knowledge from previous observations '''
                    meth = 'exploit'
                    idx  = np.argmax(out_t)       # best action
                a_t[idx] = 1                      # flag action

                '''---------------Execute action in emulator----------------'''
                x_t1, r_t, s_ep, terminal = game.step(a_t)        # take action
                x_t1 = self.resample(x_t1)              # preprocess next state
                x_t1 = self.threshold(x_t1)
                x_t1 = np.reshape(x_t1, (80, 80, 1))
                s_t1 = np.append(x_t1, s_t[:, :, :3], axis = 2)
                r_ep += r_t                               # update reward total
                
                '''---------------Store experience in memory----------------'''
                # (state, action, reward, state', terminal)
                # Deque has maxlen = self.BUF_SIZE
                self.add_exp(s_t, a_t, r_t, s_t1, terminal)

                if terminal:  # check for game over
                    s_t1 = np.zeros(s_t.shape)          # clear next state
                    self.memory.memory[-1][3] = s_t1    # update replay memory
                    self.reward_list.append((ep, r_ep)) # log episode reward
                    self.score_list.append((ep, s_ep))  # log episode score
                    print('Episode:\t{ep}\t',           # show episode stats
                          '| Score:\t{s_ep}\t'
                          '| Steps:\t{t}\t',
                          '| Reward:\t{r_ep}\t',
                          '| Loss:\t{loss}\t',
                          '| Terminal E:\t{E}',
                          '-' * 80 + '\n')
                    t = self.STEPS               # set as last step in episode
                    continue                     # skip to next episode

                '''-------Train after collecting enough observations--------'''
                if self.step > self.OBSERVE:   # total steps, not episode steps
                    E = decay(E)               # decay exploration probability

                    ''' Select random sample for training '''
                    mini = self.memory.sample(self.BATCH)
                    # Minibatch properties
                    s_j_bat  = [d[0] for d in mini]   # states
                    a_bat    = [d[1] for d in mini]   # actions
                    r_bat    = [d[2] for d in mini]   # rewards
                    s_j1_bat = [d[3] for d in mini]   # states'

                    y_bat = []   # keep track of Q values
                    out_bat = greedy(out, s, s_j1_bat)
                    for i in range(len(mini)):
                        terminal = mini[i][4]         
                        if terminal:               # check for game over
                            y_bat.append(r_bat[i]) # track reward value
                        else:   # 
                            r = r_bat[i] + self.GAMMA * np.max(out_bat[i])
                            y_bat.append(r)  # add new entry

                    # Take next step via Gradient Descent
                    play.run(feed_dict = {
                        targetQ : y_bat,
                        a : a_bat,
                        s : s_j_bat
                    }) 
                # Update values
                s_t = s_t1
                t  += 1

            # Save progress every 10000 transitions
            if t % 10000 == 0: save.save(sess, 
                                         'saved/' + self.NAME + '-dqn', 
                                         global_step = t)
            # Log state parameter values
            state = ''
            if t <= self.OBSERVE:
                state = 'observe'
            elif t > self.OBSERVE and t <= self.OBSERVE + self.EXPLORE:
                state = 'explore'
            else: state = 'train'
            self.S_LOG.write('EPISODE', ep,
                             '/ STEP', t, 
                             '/ STATE', state, 
                             '/ METHOD', meth,
                             '/ EPSILON', E,
                             '/ ACTION', idx,
                             '/ REWARD', r_t,
                             '/ MAX_Q %e' % np.max(out_t) + '\n')

            # Write to log files and save image every so often
            if t % self.EPISODES <= 100000:
                self.A_LOG.write(','.join([str(x) for x in out_t]) + '\n')
                self.H_LOG.write(','.join([str(x) for x in fc1_h.eval(\
                                 feed_dict = {s : [s_t]})[0]]) + '\n')
                cv2.imwrite('logs/frame' + str(t) + '.png', x_t1)

    def add_exp(self, s_t, a_t, r_t, s_t1, term):
        self.memory.add(s_t, a_t, r_t, s_t1, term)

    def run_emulation(self):
        deepq = DeepQ()
        s, out, fc1_h = deepq.network()
        train(s, out, fc1_h, tf.InteractiveSession())
    
    def greedy(self, out, s, s_j):
        return out.eval(feed_dict = {s : s_j})  

    def resample(self, img):
        return cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)

    def threshold(self, img):
        ret, img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
        return img

    def decay(self, E):
        # Check if network has surpassed observation period
        if E > self.TERM_EPSILON:
            E -= (self.INIT_EPSILON - self.TERM_EPSILON) / self.EXPLORE
        return E

    def loss(self, actions, out):
        # Defines the loss function of actions for state
        # Q has 2 dimensions, each corresponsding to an action
        # Set training target to max Q
        targetQ = tf.placeholder(tf.float32, [None], name = 'target')
        Q = tf.reduce_sum(tf.multiply(out, actions), axis = 1)
        loss = tf.reduce_mean(tf.square(targetQ - Q))
        return Q, targetQ, loss

    def preprocess(self, img):
        # Reshape to 80x80x4 and convert to grayscale
        img = resample(img)
        ret, img = threshold(img)
        # Generate state
        s = np.stack((img, img, img, img), axis = 2)
        return img, s

    def save_load(self, sess, load_save = False):
        save = tf.train.Saver()  
        checkpoint = tf.train.get_checkpoint_state('saved')

        if load_save:
            try:
                save.restore(sess, checkpoint.model_checkpoint_path)
                print('Successfully Loaded: ' + checkpoint.model_checkpoint_path)
            except: print ('Existing Networks Not Found')
        return save
    

#---------------------------------------------------------
# Method: get_reward()

# Establishes a penalty and reward structure for proximity
# of bird spirte to pipe sprites. Reward maximizing the
# distance, and penalize the closer bird is to a pipe
#---------------------------------------------------------
    def get_reward(self):
        reward = 0                       # rewards maximizing distance
        penalty = 0                      # penalizes getting too close to pipes
        
        #current_pos = self.sim.pose[:3]  # position has moved from the start
        #pos_dist = np.array(current_pos) - np.array(self.target_pos) # distance between current and target pos
        #e_angles = self.sim.pose[3:6]    # euler angle of each axis
        #dist = abs(pos_dist).sum()
        
        # add a penalty for euler angles at take off to steady lift
        #penalty += abs(e_angles).sum() ** 2
        
        # add a penalty for distance from target
        #penalty += dist
        
        # add reward for nearing or reaching target goal
        #if dist == 0: reward += 1000     # very large reward for reaching target
        #elif dist <= 10: reward += 500   
        #elif dist <= 50: reward += 250   # increase reward as dist gap closes
        #elif dist <= 100: reward += 100
        #else: reward += 10               # small reward for episode completion
            
#return np.tanh(reward - penalty *.005) # deduct penalties from final reward