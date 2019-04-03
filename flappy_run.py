from learner.flappy_util import Parameter, Utility

'supported agents:' 
# CustomDQ, DQN, DDPG, SARSA, CEM
'supported models:' 
# custom, VGG16, ResNet50
'supported processors:' 
# MultiInput, WhiteningNormalizer, None
'supported policy:'
 # BoltzmannQ, MaxBlotzmannQ, LinearAnnealed, Softmax, EpsGreedyQ, GreedyQ
'supported memory:' 
# RingBuffer, Sequential, EpisodeParameter
'supported optimizers:'
# Adam, Adamax, Adadelta, SGD, RMSprop
'supported loss functions:'
# there are many options, suggestions:  logcosh, mse, binary_crossentropy, poisson
inputs = {
    'game': {
        'name': 'FlappyBird',
        'fps': 30,
        'fps_tick': 2,
        'target': 40,
        'difficulty': 'hard',   # easy / medium / hard
    },
    'agent': {  
        'name': 'Custom',       # custom model: Keras CNN
        'processor': None, 
        'action_size': 2,
        'max_episodes': 20000,
        'max_time': 5,               # minutes
        'keep_gif_score': 4,
        'memory': {
            'mem_type': 'Sequential',
            'limit': 50000,
            'interval': 1,
        },
        'model': {
            'name': 'Custom',
            'state_size': 8,
            'filter_size': 64,
            'learning_rate': 0.001,
            'regulizer': 0.01,
            'alpha': 0.05,
            'gamma': 0.99,          # reward discount factor
            'momentum': 0.01,
            'decay': 0.001,
            'noise_amp': 0.1,
            'loss_function': 'logcosh',
            'optimizer': 'adadelta',
            'policy': 'BoltzmannQ',
            'test_policy': 'MaxBoltzmannQ',
            'multiprocess': True,
            'dueling_network': True,    # DQN
            'dueling_type': 'avg',      # DQN
            'double_dqn': True,         # DQN

            'train': {
                'fpa': 1,
                'begin': False,
                'batch_size': 32,
                'batch_idx': None,          # sample [idxs]. None = random
                'initial_epsilon': .2,
                'terminal_epsilon': 0.0001,
                'anneal': 500000,           # steps to cool down epsilon
                'epochs': 10,
                'split': .2,
                'validate': True,
                'shuffle': True,
                'verbose': 0,               # 0, 1, or 2
                'interval': 1,
                'warmup': 1000,
            },

            'save': {
                'path': 'saved',
                'file_name': 'flappy_model',
                'interval': 15000,
                'full': False,
                'weights': True,
                'json': True,
                'viz': True,
    }}},
    'log': {
        'path': 'logs',
        'file_type': '.json',    # .json, .csv, .tsv, .log,..., mongodb
        'log_step': True,
        'log_episode': True,
        'log_session': True,
}}

params = Parameter(inputs)