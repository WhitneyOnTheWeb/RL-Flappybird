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
agent = dqn(inputs)

'''Fit and Train model with BetaFlap Workflow'''
# this is a very memory intensive task, so its
# broken down into smaller increments
#iters = 250
#for i in range(iters):
    
#    print(
#        'Training Iterations [{}:{}]--------------------------------------------------'.\
#            format(i, i + 100)
#    )
#done = agent.fit(i, iters)
done = agent.fit()
gc.collect()
#    if done: break