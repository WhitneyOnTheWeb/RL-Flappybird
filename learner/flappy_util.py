import sys
sys.path.append('../')
sys.path.append('../game')

import os
import csv
import cv2
import math
import time
import json
import uuid
import imageio
import numpy as np
import pprint as pp
import random as rand
import tensorflow as tf
import game.flappy as flappy
import game.flappy_load as fl
from keras import backend as K
from keras.utils import plot_model, Sequence
from tensorflow import ConfigProto, Session, Graph
from learner.flappy_deep_rl import RLAgent, RLModel, Buffer

'''
Deep Reinforcement Learning: Flappy Bird with Keras
File:    flappy_util.py
Author:  Whitney King
Date:    April 2, 2019
'''

class Parameter(dict):
    def __init__(self, params, **kwargs):
        dict.__init__(params, **kwargs)
        util = Utility()
        util.display_status('Beginning RL Parameter Initialization')
        IMAGES, HITMASKS = fl.load()
        SCREEN_W = 288
        SCREEN_H = 512

        '--- Populate SubDictionaries from Inputs ----------------------------'
        self.update({
            'session': {'id': util._get_id(),
                        'log': {}},      # unique identifier
            'agent': params['agent'],
        })
        self['agent']['model']['train']['epsilon'] = \
            self['agent']['model']['train']['initial_epsilon']
        '--- Populate Parameter Dictionary for Backend -----------------------'
        self['game'].update({
            'name': params['game']['name'],
            'settings': {
                'pygame': {
                    'fps': params['game']['fps'],
                    'clock': None,
                    'tick': params['game']['fps_tick'],
                    'difficulty': params['game']['difficulty'],
                    'target': params['game']['target'],
                    'images': IMAGES,
                    'hitmasks': HITMASKS,
                },
                'screen': {
                    'icon': None,
                    'background_w': IMAGES['background'].get_width(),
                    'display': None,
                    'h': SCREEN_H,
                    'w': SCREEN_W,
                    'base_x': 0,
                    'base_y': SCREEN_H * 0.79,
                    'base_w': IMAGES['base'].get_width(),
                    'base_sft': IMAGES['base'].get_width() - \
                        IMAGES['background'].get_width(),
                },
                'player': {
                    'h': IMAGES['player'][0].get_height(),
                    'w': IMAGES['player'][0].get_width(),
                    'x': int(SCREEN_W * 0.2),
                    'y': int((SCREEN_H - \
                        IMAGES['player'][0].get_height(),) / 2),
                    'idx_gen': None,
                    'idx': 0,
                    'y_vel':-9,        # player's velocity along Y
                    'y_vel_max': 10,   # max vel along Y, max descend speed
                    'y_vel_min': -8,   # min vel along Y, max ascend speed
                    'y_acc': 1,        # players downward acceleration
                    'rot': 45,         # player's rotation
                    'rot_vel': 3,      # angular speed
                    'rot_thr': 20,     # rotation threshold
                    'flap_acc': -9,    # players speed on flapping
                    'flapped': False,  # True when player flaps
                },
                'pipe': {
                    'h': IMAGES['pipe'][0].get_height(),
                    'w': IMAGES['pipe'][0].get_width(),
                    'x_vel': -4,
                    'gap': {
                        'size': None,
                        'loc': [],
                    },
                    'upper': [],
                    'lower': [],
                },
                'track': {
                    'scored': False,
                    'score': 0,
                    'status': 'play',
                    'crash': False,
                    'loopIter': 0,
        }}})

        settings = self['game']['settings']
        player = settings['player']
        pipe = settings['pipe']
        screen = settings['screen']
        track = settings['track']

        player.update({
            'x_mid': player['x'] + (player['w'] // 2),
            'x_right': player['x'] + player['w'],
            'y_mid': player['y'] + (player['h'] // 2),
            'y_btm': player['y'] + player['h'], 
            'vis_rot': player['rot_thr']
        })

        self['session'].update({
            'graph': tf.get_default_graph(),
            'reward': [],               # rewards by episode
            'score': [],                # score by episode
            'steps': [],                # steps by episode
            'step': 0,                  # global session step number
            'start': None,              # session start time
            'end': None,                # session end time
            'elapsed': None,
            'status': 'play',
            'episode': {
                'nb': 1,                # episode counter
                'max_steps': self['game']['fps'] * self['agent']['max_time'] * 60,
                'reward': 0,            # reward for all steps in episode
                'score': 0,             # episode score
                'steps': 0,             # track number of steps in episode
                'log': [],              # append each step log
                'step': {
                    't': 0,             # step number
                    'state': None,      # preprocessed stack of frames
                    'action': np.zeros([self['agent']['action_size']]),
                    'reward': 0,        # reward for state
                    'next': None,       # next state stack
                    'terminal': False,  # is game over
                    'flap': False,      # player flapped (bool)
                    'Qs': [None, None], # predicted action probabilities 
                    'randQ': None,      # random action index
                    'maxQ': None,       # max action probability
                    'method': 'Wait',   # explore / exploit / wait
                    'index': 0,         # action selection index
                    'message': None,    # human friendly directional guidance
                    'history': None,
                    'image': {
                        'x': None,
                        'xfm': None,
                }},
                'gif': {
                        'x': [],
                        'xfm': [],
                },
            }})

        self['session']['log'].update({
            'session_log': '/sess_{}'.\
                format(self['session']['id']),
            'episode_log': '/ep{}_{}'.\
                format(self['session']['episode']['nb'],
                        self['session']['id']),
            'game_log': '/game_{}'.\
                format(self['session']['id']),
            'model_log': '/modelhist_{}'.\
                format(self['session']['id']),
        })
        self['session']['log'].update(params['log'])

        config = ConfigProto()
        config.gpu_options.allow_growth = True

        self['session']['config'] = config
        self['session']['activate'] =\
            Session(config=self['session']['config'], 
                    graph=self['session']['graph'])

        util.display_status('Hyperparameters Successfully Loaded')
        
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)


class Utility:
    def __init__(self):
        super(Utility, self).__init__()

    def _get_timestamp(self):
        return time.strftime('%Y-%m-%d@%H:%M:%S')

    def _get_id(self):
        return time.strftime('%d%m%Y%H%M%S')

    def display_status(self, status):
        print('{} | {}...'.format(self._get_timestamp(), status))

    def initialize_rl_session(self, params):
        '--- Game Environment and Parameters' 
        game = params['game']
        game['env'] = flappy.Environment(
            target_score=game['target'],
            difficulty=game['difficulty'],
            fps=game['fps']
        )
        
        # GAME SETTINGS

        self.display_status('{} Environment Initialized'.\
            format(game['name']))

        '--- Allocate Experience Replay Memory'
        agent = params['agent']
        memory = agent['memory']
        memory['store'] = Buffer(memory['limit'], 
                                 memory['mem_type'], 
                                 window_length=1)
        self.display_status('{} Replay Cache Initialized'.\
            format(memory['mem_type']))

        '--- Create Agent Worker'
        agent['object'] = RLAgent(params)
        self.display_status('{} Agent Initialized'.\
            format(agent['name'].capitalize()))

        '--- Activate Keras GPU Session'
        session = params['session']
        K.set_session(session['activate'])
        self.display_status('Keras GPU Session {} Beginning'.\
            format(session['id']))

        '--- Create Keras Model Worker'
        model = agent['model']
        model.update({
            'worker': RLModel(
                        model=model['name'],
                        A=agent['action_size'],
                        S=model['state_size'],
                        H=model['filter_size'],
                        lr=model['learning_rate'],
                        alpha=model['alpha'],
                        reg=model['regulizer'],
                        momentum=model['momentum'],
                        decay=model['decay'],
                        loss=model['loss_function'],
                        opt=model['optimizer']
        )})
        self.display_status('{} Keras Model Compiled'.\
            format(model['name']))
        
        '--- Load Saved Model Weights'
        worker = model['worker'].nn
        save = model['save']
        load_path = save['path']
        file = save['file_name']
        try:  
            worker.load_weights(load_path + '\\' + file + '_weights.h5')
            self.display_status('Saved Keras Model Weights Loaded')
        except:
            self.display_status('No Saved Keras Model Weights Found')

    def log_session(self, params):
        '''---Log session information---'''
        session = params['session']
        log = session['log']
        keys = ['id', 'start', 'end', 'elapsed', 'steps']
        data = { x : session[x] for x in keys }
        data['episodes'] = params['agent']['max_episodes']
        self._write_log(session, log, 'session', data)

    def log_episode(self, params):
        '''---Log episode information---'''
        session = params['session']
        episode = session['episode']
        log = session['log']
        keys = ['nb', 'reward', 'score', 'steps', 'log']
        data = { x : episode[x] for x in keys }
        data['sess'] = session['id']
        self._write_log(session, log, 'episode', data)
    
    def log_step(self, params):
        '''---Log agent step information---'''
        agent = params['agent']
        session = params['session']
        log = params['session']['log']
        episode = session['episode']
        step = episode['step']
        image = step['image']
        keys = ['t', 'action', 'reward', 'flap', 'Qs', 
                'randQ', 'maxQ', 'method', 'message', 'history']
        data = { x : step[x] for x in keys }
        data['game_log'] = self.log_game(params)
        data['image'] = image['x']
        return data

    def log_game(self, params):
        '''---Log game step information---'''
        game = params['game']
        log = params['session']['log']
        keys = ['player', 'pipes', 'terminal', 'scored', 'score']
        data = { x : game[x] for x in keys }
        return data

    def log_state(self, params):
        '''---Log model state information---'''
        agent = params['agent']
        session = params['session']
        log = params['session']['log']
        episode = session['episode']
        step = episode['step']
        keys = ['state', 'action', 'reward', 'next', 'terminal']
        data = { x : step[x] for x in keys }
        data['training'] = agent['model']['train']['begin']
        self._write_log(session, log, 'state', data)

    def log_hist(self, params):
        '''---Log model state information---'''
        session = params['session']
        episode = session['episode']
        step = episode['step']
        log = params['session']['log']
        data = step['history']
        self._write_log(session, log, 'history', data)
    
    def _write_log(self, session, log, ltype, data):
        ftype = log['file_type']
        name = '/' + ltype + '_' + session['id'] + '.'
        path = log['path'] + name + ftype

        '--- Add data to log'
        if ftype == '.json':  # dump data in json file 
            with open(path, 'a+') as f: json.dump(data, f)
        else:                 # log to standard flat file
            with open(path, 'a+', newline='') as f:
                writer = csv.DictWriter(f, data.keys())
                if os.stat(path).st_size == 0: writer.writeheader()
                writer.writerow(data)
        self.display_status('{} Logged in {}'.format(name.capitalize(), path))
        f.close()
        #if ftype == 'mongodb': pass  # stretch goal, hook to flappy_freeze.py

    def save_model(self, params):
        agent = params['agent']
        model = agent['model']
        worker = model['worker'].nn
        save = model['save']
        path = save['path'] + '/' + save['file_name']

        '''---Save full model to single .h5 file---'''
        if save['full']:
            worker.save(path + '_full.h5', overwrite=True)
            self.display_status('{} Model Saved to {}'.\
                format(model['name'], path + '_full.h5'))
        '''---Save model weights to separate .h5 file---'''        
        if save['weights']:
            worker.save_weights(path + '_weights.h5', overwrite=True)
            self.display_status('{} Model Weights Saved to {}'.\
                format(model['name'], path + '_weights.h5'))
        '''---Save model structure as JSON file---'''
        if save['json']:
            with open(path + '.json', 'w+') as f:
                json.dump(worker.to_json(), f)
            f.close()
            self.display_status('{} Model Structure Saved to {}'.\
                format(model['name'], path + '.json'))
        '--- Save diagram of model neural network logic'
        if save['viz']:
            plot_model(worker, to_file=path + '_flow.png')
            self.display_status('{} Neural Network Diagram Saved to {}'.\
                format(model['name'], path + '_flow.png'))

    def preprocess_input(self, x):
        '''---Preprocess frames for neural network---
            * Reorient and esize image [512 x 288] -> [80 x 80] 
            * Convert to grayscale '''
        x = cv2.transpose(x)  # (Y, X) --> (X, Y)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, (80, 80))
        x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)[1]
        return x

    def create_episode_gif(self, params):
        '''---Save GIF of Episodes with High Enough Scores---'''
        session = params['session']
        log = session['log']
        episode = session['episode']
        gif = episode['gif']

        self.display_status('Saving Episode {} as GIFs'.\
            format(episode['nb']))
        x_path = '/gifs/xep{}_{}.gif'.\
            format(episode['nb'], session['id'])
        xfm_path = log['path'] + '/gifs/xfmep{}_{}.gif'.\
            format(episode['nb'], session['id'])
        imageio.mimsave(x_path, gif['x'])
        imageio.mimsave(xfm_path, gif['xfm'])
        self.display_status('GIFs saved successfully')

    def create_state(self, x, params):
        S = params['agent']['model']['state_size']
        session = params['session']
        episode = session['episode']
        step = episode['step']
        t = step['t']
        s_t = step['state']

        if t == 0:
            x = np.stack([x] * S, axis=2)
        else:
            x = np.reshape(x, (80, 80, 1))  # channels dimension
            x = np.append(s_t[0, :, :, 1:], x, axis=2)
        x = np.reshape(x, (1, *x.shape))
        return x

    def get_action(self, x, params):
        ''' Observation period occurs once, spans across episodes
            * Fills replay memory with random training data
            * After observation, training begings, epsilon anneals
            * Determines Exploration or Exploitation probability
        '''
        session = params['session']
        step = session['episode']['step']
        agent = params['agent']
        model = agent['model']
        train = model['train']
        A = agent['action_size']
        E = train['epsilon']
        a_t = np.zeros(A)
        Qs = np.zeros(A)
        randQ = rand.randrange(A)
        method = 'Wait'
        idx = 0

        with session['activate'].as_default():
            Qs = model.predict(x)
        maxQ = np.argmax(Qs)                 # exploit action: idx of maxQ

        if (rand.random() <= E) or (session['step'] <= train['warmup']):
            '''Explore if rand <= Epsilon or Observing'''
            method = 'Explore'               # always random if observing
            idx = randQ
        else:
            '''---Follow Greedy Policy for maxQ values---'''
            method = 'Exploit'  # prob of predicting Q goes up as E anneals
            idx = maxQ

        a_t[idx] = 1   # flag action idx
        flap = False if a_t[0] else True

        '''---Decay Future Exploration Probability---'''
        if E > train['terminal_epsilon']:
            E = E - (train['initial_epsilon'] - train['terminal_epsilon']) \
                   / train['anneal']

        train.update({ 'epsilon': E })
        step.update({
            'action': a_t,
            'Qs': Qs,
            'randQ': randQ,
            'maxQ': maxQ,
            'method': method,
            'index': idx,
            'flap': flap,
        })

    def get_reward(self, game, step):
        # step, score, player_x, player_y, player_y_mid, player_x_mid, pipe1, pipe2, pipe_gaps, terminal
        t = step['t'] 
        score = game['score']
        scored = game['scored']
        target = game['target']
        terminal = game['terminal']
        player = game['player']
        up = game['pipes']['upper']
        low = game['pipes']['lower']
        gaps = game['pipes']['gaps'] 
        
        '--- Reward player positioning if in gap between pipe corners'
        '--- Check player position against corners of pipes[0]'
        '--- Check if player in left safe zone'
        if player['x'] < up[0]['left'][0] \
                and player['y'] > up[0]['left'][1]:
            u1 = True  # player Sw of upper pipe[0] left corner
        if player['x_right'] < low[0]['left'][0] \
                and player['y_btm'] < low[0]['left'][1]:
            l1 = True  # player Nw of lower pipe[0] left corner
        '--- Check if player in right safe zone'
        if player['x'] > up[0]['right'][0] \
                and player['y'] > up[0]['right'][1]:
            u2 = True  # player SE of upper pipe[0] right corner
        if player['x_right'] > low[0]['right'][0] \
                and player['y_btm'] < low[0]['right'][1]:
            l2 = True  # player NE of lower pipe[0] right corner
        '--- Check player position against corners of pipes[1]'
        '--- Check if player in left safe zone'
        if player['x'] < up[1]['left'][0] \
                and player['y'] > up[1]['left'][1]:
            u3 = True  # player Sw of upper pipe[1] left corner
        if player['x_right'] < low[1]['left'][0] \
                and player['y_btm'] < low[1]['left'][1]:
            l3 = True  # player Nw of lower pipe[1] left corner
        '--- Check if player in right safe zone'
        if player['x'] > up[1]['right'][0] \
                and player['y'] > up[1]['right'][1]:
            u4 = True  # player SE of upper pipe[1] right corner
        if player['x_right'] > low[1]['right'][0] \
                and player['y_btm'] < low[1]['right'][1]:
            l4 = True  # player NE of lower pipe[1] right corner

        '--- Check which gap the player is aiming for'
        if up[0]['x_right'] < player['x_mid']:
            gap = gaps[1]     # past first pipes
            pipe = up[1]
        else:                 # first pipes
            gap = gaps[0]
            pipe = up[0]

        '--- Check if player is in pipe gap'
        in_gap, gap_x, gap_y = False, False, False
        if gap['btm'] > player['y_btm'] > player['y'] > gap['top']:
            gap_y = True                  # level w/ pipe gap
            level = True
            if pipe['x'] < player['x_mid'] < pipe['x_right']:
                gap_x = True              # in pipe gap
        if gap_x and gap_y:
            in_gap = True  # flag in gap for reward

        '--- Check if player is in the danger zone'
        danger = False
        if (not u1 and not l1) and player['x'] < pipe['x']:
            danger = True
        elif (not u2 and not l2 and not u3 and not l3) \
                and (player['x'] > pipe['x_right'] or player['x'] < pipe['x']):
            danger = True
        elif (not u4 and not l4) and player['x'] > pipe['x_right']:
            danger = True

        '--- Reward scales up with steps/score/gap distance'
        step_dist = t * (1 + score)
        tar_delta = target - score

        tar_dist = np.sqrt(np.abs((-target ** 2 - score ** 2) *
                                  np.sqrt(step_dist * t)) // 4)
        gap_dist = np.sqrt((pipe['x_mid'] - player['x_mid']) ** 2
                           + (gap['mid'] - player['y_mid']) ** 2)

        reward = step_dist * .001  # base reward scales larger
        if step < 50: mul_penalty = ((100 - step) ** 2 // 2)
        else: mul_penalty = 1

        '--- Player is passing through a set of pipes'
        if in_gap:
            reward = reward + (
                np.sqrt(tar_dist) // (1 + gap_dist * (tar_delta ** 2))
            )
            msg = 'Great!'
            if scored:    # multiplier for scoring
                reward = reward + tar_dist
                msg = 'You scored!'
        # safe zone before/between pipes
        elif (u1 and l1) or (u2 and l2 and u3 and l3) or (u4 and l4) or level:
            reward = reward + (
                np.sqrt(tar_dist) // (1 + (gap_dist ** 2) * (tar_delta ** 2))
            )
            msg = 'Safe zone!'
        elif danger:
            penalty = (np.sqrt(np.abs(gap_dist ** 2 - tar_delta ** 2))
                       // (step_dist * .001)) * mul_penalty
            reward = reward - penalty
            msg = 'Danger zone!'
        elif terminal:
            penalty = 1 + np.abs(gap_dist * tar_delta * 0.1) * mul_penalty
            reward = reward - penalty
            msg = 'Boom!'
        else: msg = 'Keep going!'    # no modifier for anything unidentified

        '''Scale and constrain reward values, save as step reward'''
        '''---Hyperbolic Tangent of Reward---'''
        reward = np.tanh(reward * .0001)
        step.update({ 
            'reward': reward,
            'terminal': terminal,
            'message': msg,
        })

    def get_replay(self, step, train):
        state = (
            step['state'], 
            step['action'], 
            step['reward'], 
            step['terminal'], 
            train['begin']      # True if training
        )
        return state

    def train_model(self, params):
        '''---Select Random Batch of Experience for Training---'''
        session = params['session']
        episode = session['episode']
        step = episode['step']
        agent = params['agent']
        model = agent['model']
        memory = agent['memory']
        train = memory['train']
        store = memory['store']
        worker = model['worker'].nn

        bat = store.sample(batch_size=train['batch_size'], 
                           batch_idx=train['batch_idx'])

        s_bat = np.array([e[0][0, :, :, :, ] for e in bat])   # states
        a_bat = np.array([e[1] for e in bat])                 # actions
        r_bat = np.array([e[2] for e in bat])                 # rewards
        s1_bat = np.array([e[3][0, :, :, :, ] for e in bat])  # nexts
        term_bat = np.array([e[4] for e in bat])              # terminal

        '''---Check for Terminal, Discount Future Rewards---'''
        with session['activate'].as_default():
            tQ_bat = worker.predict(
                        x=s_bat,             # targetQ
                        batch_size=train['batch_size'],
                     )
        with session['activate'].as_default():       
            Q_bat = worker.predict(
                        s1_bat,              # futureQ
                        batch_size=train['batch_size'],
                    )          

        '''---Logic to Update maxQ Values'''
        for i, x in enumerate(term_bat):
            '''---tarQ_bat[i] = -1 if Terminal---'''
            tQ_bat[i, [np.argmax(a_bat[i])]] = (-1) if x else \
                r_bat[i] * model['gamma'] * np.max(Q_bat[i])

        '''---Train using gradient decent, loss, and Adam---
            * Fit model with training data (x) and targets (y)'''
        if train['validate']: split = train['split']
        else: split = 0
        with session['activate'].as_default():
            hist = worker.fit(
                        x=s_bat, 
                        y=tQ_bat,
                        batch_size=train['batch_size'],
                        epochs=train['epochs'],
                        verbose=train['verbose'],
                        validation_split=split,
                        shuffle=train['shuffle'],
                    )
        step['history'] = hist


    def game_over(self, params):
        '''---Halt play, display stats, end current episode'''
        agent = params['agent']
        session = params['session']
        episode = session['episode']
        step = episode['step']

        ep_stats = 'Ep: {:<8}| Step: {:<8}| Reward: {:<14.5f}| Score: {}'.\
            format(episode['nb'], step['t'], step['reward'], episode['score'])
        self.display_status(ep_stats)
        step['t'] = episode['max_steps']     # set as last step in episode

        step['next'] = np.zeros(step['next'].shape)     # clear next state
        session.update({                                # log episode stats
            'reward': session['reward'].\
                append([episode['nb'], episode['reward']]),
            'score': session['score'].\
                append([episode['nb'], episode['score']]),
            'steps': session['steps'].\
                append([episode['nb'], episode['steps']])
        })            # log episode

        if episode['score'] >= agent['keep_gif_score']:
            self.create_episode_gif(params)
        self.log_episode(params)

    def end_session(self, params):
        session = params['session']
        game = params['game']

        '''---Exit PyGame and Close Session After Last Episode---'''
        self.display_status('Training Session Complete!')
        game.quit_game()                    # quit pygame
        session['end'] = time.time()
        session['elapsed'] = time.gmtime(session['end'] - session['start'])
        session['elapsed'] = time.strftime('%H Hours %M Minutes %S Seconds',
                                           session['elapsed'])
        self.log_session(params)               # track session information
        self.display_status('Elapsed Time: {}'.format(session['elapsed']))
        print('  ___                   ___')
        print(' / __|__ _ _ __  ___   / _ \__ ______ _')
        print('| (_ / _` | ''   \/ -_) | (_) \ V / -_) ''_|')
        print(' \___\__,_|_|_|_\___|  \___/ \_/\___|_|')