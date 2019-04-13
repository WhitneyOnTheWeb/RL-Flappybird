import sys

sys.path.append('../')
sys.path.append('./game')

import gc
import numpy as np
import pprint as pp
import tensorflow as tf
import keras.backend as K
from flappy_util import Utility
from flappy_inputs import Inputs
from flappy_beta import BetaFlapDQN, Buffer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


inputs = Inputs()
inputs = inputs.params
util = Utility()
gpu = '/job:localhost/replica:0/task:0/device:GPU:0'

with tf.device(gpu):
    buffer = Buffer(                       # initialize buffer outside of agent
        limit=inputs['memory']['limit'], 
        window_length=inputs['memory']['state_size'],
    )
util.display_status(
    'Built Replay Buffer Limited to {} States'.format(inputs['memory']['limit'])
)

'''Fit and Train model with BetaFlap Workflow'''
# this is a very memory intensive task, so its
# broken down into smaller increments
iters = 20
for i in range(1, iters):
    '''---Session Parameters---'''
    sess_id = util.get_id()
    sess = util.config_session()
    with tf.device(gpu):
        K.set_session(sess)
    print(        
    'Training Iterations [{}:{}]------------------------------------------------------'.\
            format(i, i + iters-1)
    )
    with sess.as_default():
        agent = BetaFlapDQN(inputs, buffer, sess_id, sess)
        with tf.device('GPU:0'):
            done = agent.fit(i, i + iters - 1)
        buffer = agent.memory   # pass memory back to save interval over interval
        gc.collect()
        if done:
            break