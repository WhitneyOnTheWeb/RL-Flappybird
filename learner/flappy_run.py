import sys

sys.path.append('../')
sys.path.append('/game')
sys.path.append('/learner')

import gc
import numpy as np
import pprint as pp
from flappy_inputs import Inputs
from flappy_beta import BetaFlapDQN as dqn


inputs = Inputs()
inputs = inputs.params

'''Fit and Train model with BetaFlap Workflow'''
# this is a very memory intensive task, so its
# broken down into smaller increments
for _ in range(250):
    agent = dqn(inputs)
    print(
        'Training Iterations [{}:{}]--------------------------------------------------'.\
            format(_, _ + 100)
    )
    agent.fit(_)
    gc.collect()