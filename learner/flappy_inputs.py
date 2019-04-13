import numpy as np

class Inputs:
    params = {
        'game': {
            'name': 'FlappyBird',         # name of gmae being played
            'fps': 30,                    # frames per second
            'tick': 4,                    # clock ticks per second
            'target': 40,                 # target score
            'difficulty': 'hard',         # gap size [easy, medium, hard]
        },
        'agent': {
            'name': 'DeepQ',              # name of the agent
            'action_size': 2,             # number of possible actions
            'delta_clip': np.inf,         # constrain reward range
            'session': {
                'max_ep': 1000,           # max episodes to play
                'episode': {
                    'max_time': 5,        # max minutes per episode
                    'keep_gif_score': 4,  # save gif of episode
        }}},
        'model': {
            'type': 'Custom',             # neural net architecture
            'filter_size': 32,           # filters on state
            'optimizer': 'adadelta',      # adam, adamax, adadelta, rmsprop, sgd
            'regulizer': 0.01,
            'alpha': 0.025,               # test value
            'gamma': 0.99,                # reward discount factor
            'momentum': 0.01,             # SGD 
            'decay': 0.001,               # SGD
            'noise_amp': 0.1,       
            'target_update': 1,           # update model targets interval
            'dueling_network': False,     # DQN - model weights will not save correctly if enabled
            'dueling_type': 'max',        # DQN
            'double_dqn': True,           # DQN
            'training': { 
                'verbose': 2,             # 0, 1, or 2
                'interval': 1,            # train every n steps
                'action_repetition': 1,   # frames per action
                'warmup': 1000,           # observation steps
                'max_ep_observe': 40,     # episode random seed (<=40)
                'learn_rate': .1,         # optimizer learn rate
                'initial_epsilon': .1,    # probability of random action
                'terminal_epsilon': 0.0001,
                'anneal': 50000,          # steps to cool down epsilon
                'epochs': 1,              # training iterations on batch
                'split': .1,              # train/test split
                'validate': True,        # split and validate batch
                'shuffle': True,         # shuffle batch
                'training': False,        # force force training
            },
            'save': {
                'save_n': 1000,           # save model steps interval
                'log_n': 1,               # log model steps interval
                'save_path': 'saved',     # model save location
                'log_path': 'logs',       # model log location
                'ftype': '.json',         # model log type
                'save_full': True,        # full model
                'save_weights': True,     # model weights
                'save_plot': True,        # neural net diagram
                'save_json': True,        # model json
                # if verbose == 1: TrainIntervalLogger(interval = log_n)
                # if verbose > 1: TrainEpisodeLogger()
                'verbose': 1,             # Save / Logging Callback !! NOT IMPLEMENTED
                'visualize': False,       # Visualizer() Callback  !! NOT IMPLEMENTED
        }},
        'memory': {
            'state_size': 8,             # observations to stack for state
            'batch_size': 128,            # states in a training batch
            'batch_idx': None,            # specific memory index !! NOT IMPLEMENTED
            'limit': 100000,              # max entries in memory                      
            'interval': 100               # save step to memory every n
}}