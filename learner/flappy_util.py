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
import matplotlib.image as img
import matplotlib.pyplot as plt
import keras.backend as K

class Utility:
    def get_timestamp(self):
        return time.strftime('%Y-%m-%d@%H:%M:%S')

    def get_id(self):
        return time.strftime('%d%m%Y%H%M%S')

    def get_log_dir_struct(self, sid, ldir, ftype):
        name = '/session_{}'.format(sid)
        path = '/learner/' + ldir + name + ftype
        return path

    def get_save_dir_struct(self, sdir, name):
        path = sdir + '/' + name
        return path        

    def display_status(self, status):
        print('{} | {}...'.format(self.get_timestamp(), status))  

    def mean_q(self, y_true, y_pred):
        return K.mean(K.max(y_pred, axis=-1))
    
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
        
    '''
    Refrence: Keras RL Visualize Log Example
    https://github.com/keras-rl/keras-rl/blob/master/examples/visualize_log.py
    '''
    def visualize_log(self, filename, figsize=None, output=None):
        with open(filename, 'r') as f:
            data = json.load(f)
        if 'episode' not in data:
            raise ValueError(
                'Log file "{}" does not contain the "episode" key.'.\
                    format(filename)
            )
        episodes = data['episode']
        # Get value keys. The x axis is shared and is the number of episodes.
        keys = sorted(list(set(data.keys()).difference(set(['episode']))))
        if figsize is None:
            figsize = (15., 5. * len(keys))
        f, axarr = plt.subplots(len(keys), sharex=True, figsize=figsize)
        for idx, key in enumerate(keys):
            axarr[idx].plot(episodes, data[key])
            axarr[idx].set_ylabel(key)
        plt.xlabel('episodes')
        plt.tight_layout()
        if output is None: plt.show()
        else: plt.savefig(output)


    def pickle_to_json(self, pkl):
        with open(pkl, 'rb') as p:
            json_pkl = jsonpickle.decode(pkl)
        p.close()

        fname = pkl.split('.')[0]
        with open(fname + '.json', 'a+') as j:
            json.dumps(json_pkl)
        j.close()




        


