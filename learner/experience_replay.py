from collections import namedtuple, deque
import time
import random
import numpy as np
'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    experience_replay.py
Author:  Whitney King
Date:    March 9, 2019

References:
    Udacity ML Engineer Nanodegree Classroom
    tinyurl.com/yd7rye3w
'''

class Buffer:
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        # Memory Stored as Deque
        self.memory = deque(maxlen=buffer_size)
        print('{} | Experience Replay Memory Initialized...'.format(self.timestamp()))

    def timestamp(self): return time.strftime('%Y-%m-%d@%H:%M:%S')
    
    def add(self, s, a, r, s_t, done):
        """Add a new experience to memory."""
        e = (s, a, r, s_t, done)
        self.memory.append(e)

    def sample(self, batch_size = 32):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def reset(self):
        """Clear all experiences in memory"""
        self.memory.clear()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)