from collections import namedtuple, deque
import random
import numpy as np
import copy
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
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", 
                                                  "action", 
                                                  "reward", 
                                                  "next_state", 
                                                  "done"])

    def add(self, s, a, r, s_t, done):
        """Add a new experience to memory."""
        e = self.experience(s, a, r, s_t, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k = self.batch_size)

    def reset(self):
        """Clear all experiences in memory"""
        self.memory.clear()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)