import os
import sys
import csv
import cv2
import time
import json
import pickle
import imageio
import jsonpickle
import numpy as np
import pprint as pp
import random as rand
from learner.flappy_util import Utility

'---Keras / Tensorflow Modules'
import tensorflow as tf
from keras import metrics
import keras_metrics as km
from keras import backend as K
from rl.callbacks import ModelIntervalCheckpoint, Visualizer, Callback, CallbackList
from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger
from keras.callbacks import History, ModelCheckpoint, TensorBoard
from keras.callbacks import Callback as KerasCallback
from keras.callbacks import CallbackList as KerasCallbackList

class FlappySession(Callback):
    def __init__(self):
        super(FlappySession, self).__init__()
        self.util = Utility()
        self.starts = {}
        self.ends = {}
        self.duration = {}

    def on_train_begin(self, logs={}):
        self.log_episodes = []
        self.starts['session'] = time.time()   
        
    def on_train_end(self, logs, sess_id, path):
        '--- Add data to log'
        self.ends['session'] = time.time()
        self.duration['session'] = time.gmtime(
            self.ends['session'] - self.starts['session'])
        self.elapsed = time.strftime(
            '%H Hours %M Minutes %S Seconds', self.duration['session'])
        
        logs.update({
            'start': self.starts,
            'end': self.ends,
            'duration': self.duration,
            'episodes': self.log_episodes,
        })

        '''---Exit PyGame and Close Session After Last Episode---'''
        self.util.display_status('Training Session Complete!')
        self.util.display_status('Elapsed Time: {}'.format(self.elapsed))
        print('  ___                   ___')
        print(' / __|__ _ _ __  ___   / _ \\__ ______ _')
        print('| (_ / _` | ''   \\/ -_) | (_) \\ V / -_) ''_|')
        print(' \\___\\__,_|_|_|_\\___|  \\___/ \\_/\\___|_|')
        self.util.display_status(
            'Writing Session {} Log to JSON File'.format(sess_id)
        )

        def default(o):
            if isinstance(o, np.int64): return int(o)  

        with open(os.getcwd() + path, 'a+') as f: 
            json.dump(
                logs, 
                f, 
                sort_keys=True, 
                indent=4, 
                default=default,
            )
        f.close()
        self.util.display_status(
            'Session {} Logged in {}'.format(sess_id, path)
        )
        

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        self.total_loss = []
        self.log_steps = []
        self.frames = []                         # images of each frame
        self.t_frames = []
        self.starts['episode'] = time.time()

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        self.ends['episode'] = time.time()
        self.duration['episode'] = self.ends['episode'] - self.starts['episode']

        '''---Halt play, display stats, end current episode'''
        ep_stats = 'Iter: {} | Game: {:<5}| Steps: {:<6}| Rwd: {:<13.5f}| Score: {:<2}'.\
            format(
                logs['iteration'], 
                logs['episode'], 
                logs['steps'], 
                logs['reward'], 
                logs['score']
            )
        self.util.display_status(ep_stats)
        logs.update({
            'start': self.starts['episode'],
            'end': self.ends['episode'],
            'durations': self.duration['episode'],
        })
        self.log_episodes.append(logs)
        self.log_episodes.append(self.log_steps)

        sid = logs.pop('sess_id')

        if logs['score'] >= logs['gif']:
            '''---Save GIF of Episodes with High Enough Scores---'''
            self.util.display_status(
                'Saving Episode {} as GIF'.format(logs['episode'])
            )
            path = os.getcwd() + '\\logs\\gifs\\'
            x_path = path + 'xep{}_{}.gif'.\
                format(logs['episode'], sid)
            xfm_path = path + 'xfmep{}_{}.gif'.\
                format(logs['episode'], sid)
            imageio.mimsave(x_path, self.frames)
            imageio.mimsave(xfm_path, self.t_frames)
            self.util.display_status('GIFs saved successfully')

        # Write episode log to pickle file
        #path = os.getcwd() + '\\logs\\'
        #if logs['episode'] % 20 == 0:
        #    pkl_path = path + '/steps_{}.pkl'.format(sid)
        #    with open(pkl_path, 'wb+') as f:
        #        pickle.dump(self.log_episodes, f)
        #    f.close()

    def on_step_begin(self, step):
        """Called at beginning of each step"""
        self.starts['step'] = time.time()
        

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        self.ends['step'] = time.time()
        self.duration['step'] = self.ends['step'] - self.starts['step']
        self.frames.append(logs.pop('x'))        # save image of frame
        self.t_frames.append(logs.pop('x_t'))  # save transformed frame

        logs.update({
            'start': self.starts['step'],
            'end': self.ends['step'],
            'durations': self.duration['step'],
        })
        self.log_steps.append(logs)
        
        
