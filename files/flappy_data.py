import os
import sys
import time
import json
import uuid
import pickle
import pymongo
import numpy as np
from collections import defaultdict


class DeepQDB:
    def connect(db_name = 'flappy'):
        db_client = pymongo.MongoClient()       # connect to MongoDB
        db = db_client[db_name]                 # load database
        conn = db['sessions']                   # connect to collection
        return conn

    def insert(log):
        conn = DeepQDB.connect()
        conn.insert_one(log)
    
    def episode_template(ep, max_steps):
        '''---Template defining format for logged episde information---'''
        node = {
            'id': str(uuid.uuid1()),
            'number': int(ep),
            'begin': time.strftime('%Y-%m-%d@%H:%M:%S'),
            'end': None,
            'score': 0,
            'reward': None,
            'n_steps': None,
            'frame_action': None,
            'max_steps': int(max_steps),
            'steps': [None],                  # array of template_steps()
        }
        return node

    def step_template(step):
        '''---Template defining format for logged step information---'''
        node = {
            'id': str(uuid.uuid1()),
            'number': int(step),
            'state': None,                    # template_state()
            'epsilon': None,
            'qs': [None],
            'max_q': None,
            'rand_q': None,
            'method': None,
            'message': None,
            'frame': None,
            'trans_frame': None
        }
        return node

    def state_template(s, a, r, st, term):
        '''---Template defining format for logged state information---'''
        node = {
            'id': str(uuid.uuid1()),
            'state': s.tolist(),
            'action': a.tolist(),
            'reward': float(r),
            'next': st.tolist(),
            'terminal': term,
        }
        return node

    def session_template(game, max_ep, obsrv, explr,  e_init, e_term, 
                         tar, gam, lr, s, a, h, buff, bat, sess_id):
        '''---Template defining format for logged session information---'''
        node = {
            'id': str(sess_id),
            'game': {
                'name': str(game.name),
                'fps': int(game.fps),
                'difficulty': str(game.mode),
                },
            'buffer': {
                'id': str(uuid.uuid1()),
                'exp': [None]                  # array of template_state()
                },
            'episode': [None],                # array of episode_template()
            'begin': time.strftime('%Y-%m-%d@%H:%M:%S'),
            'end': None,
            'max_eps': int(max_ep),
            'observe': int(obsrv),
            'explore': int(explr),
            'e_init': float(e_init),
            'e_term': float(e_term),
            'target': int(tar),
            'gamma': float(gam),
            'learning_rate': float(lr),
            'mem_size': int(buff),
            'batch_size': int(bat),
            'state_size': int(s),
            'action_size': int(a),
            'filter_size': int(h)
        }
        return node