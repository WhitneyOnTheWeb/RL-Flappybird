import numpy as np

class Inputs:
    params = {
        'game': {
            'name': 'FlappyBird',
            'fps': 30,
            'tick': 4,  
            'target': 40,
            'difficulty': 'hard',  
        },
        'agent': {
            'name': 'DeepQ',
            'action_size': 2,
            'delta_clip': np.inf, 
            'session': {
                'max_ep': 1000,  
                'episode': {
                    'max_time': 5,              # minutes
                    'keep_gif_score': 4,
        }}},
        'model': {
            'type': 'Custom',
            'filter_size': 64,
            'optimizer': 'adadelta',
            'regulizer': 0.01,
            'alpha': 0.05,
            'gamma': 0.99,          # reward discount factor
            'momentum': 0.01,
            'decay': 0.001,
            'noise_amp': 0.1, 
            'target_update': 100,
            'dueling_network': False,    # DQN
            'dueling_type': 'max',      # DQN
            'double_dqn': False,        # DQN
            'training': { 
                'verbose': 0,               # 0, 1, or 2
                'interval': 1,
                'action_repetition': 4,
                'warmup': 100,
                'max_ep_observe': 40,      # ep random start steps
                'learn_rate': 0.001,
                'initial_epsilon': 1,
                'terminal_epsilon': 0.001,
                'anneal': 20000,           # steps to cool down epsilon
                'epochs': 1,
                'split': .1,
                'validate': True,
                'shuffle': True,
                'training': False,
            },
            'save': {
                'save_n': 1000,
                'log_n': 1,
                'save_path': 'saved',
                'log_path': 'logs',
                'ftype': '.json', 
                'save_full': True,
                'save_weights': True,
                'save_plot': True,
                'save_json': True,
                # if verbose == 1: TrainIntervalLogger(interval = log_n)
                # if verbose > 1: TrainEpisodeLogger()
                'verbose': 1,         # Save / Logging Callback
                'visualize': True,   # Visualizer() Callback
        }},
        'memory': {
            'state_size': 16,
            'batch_size': 75,
            'batch_idx': None,
            'limit': 50000,   
            'interval': 1,
}}