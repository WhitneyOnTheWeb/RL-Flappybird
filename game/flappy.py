
'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    flappy.py
Author:  Whitney King
Date:    March 8, 2019

Modified Python clone of Flappy Bird game, altered
with a Deep-Q RL agent in mind. Serves as the TASK
for the AGENT to carry out in this project.

References:
    FlapPy Bird
    Author: Sourabh Verma
    github.com/sourabhv/FlapPyBird
'''

import sys
import pygame
import pygame.locals
import pygame.surfarray as surfarray
import numpy as np
import pprint as pp
import random as rand
from pygame.locals import *
from itertools import cycle
import game.flappy_load as fl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Graphic Settings
SCREEN_W = 288
SCREEN_H = 512

pygame.init()
FPSCLOCK  = pygame.time.Clock()
SCREEN    = pygame.display.set_mode((SCREEN_W, SCREEN_H))

# Obstacle Settings
BASE_Y   = SCREEN_H * 0.79 # amount base can shift left
IMAGES, HITMASKS = fl.load()

# Calculate Asset Sizes
PLAYER_W     = IMAGES['player'][0].get_width()
PLAYER_H     = IMAGES['player'][0].get_height()
PIPE_W       = IMAGES['pipe'][0].get_width()
PIPE_H       = IMAGES['pipe'][0].get_height()
BACKGROUND_W = IMAGES['background'].get_width()
BACKGROUND_H = IMAGES['background'].get_height()
BASE_W       = IMAGES['base'].get_width()

# change player respawn x every 5th iteration
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

def render(mode='human'):
    pygame.init()
    FPSCLOCK  = pygame.time.Clock()
    SCREEN    = pygame.display.set_mode((SCREEN_W, SCREEN_H))

class Environment:
    def __init__(self, target_score = 40, difficulty = 'hard', fps = 30, tick =  2):
        '''---Initialize New Game---'''        
        render()
        
        self.name = 'FlappyBird'
        self.icon = pygame.image.load('game/flappy.ico')
        self.width = SCREEN_W
        self.height = SCREEN_H
        self.fps = fps
        self.mode = difficulty
        self.tick = tick
        self.t = 0

        pygame.display.set_caption(self.name)
        pygame.display.set_icon(self.icon)

        self.crash, self.scored = False, False
        self.score, self.reward, self.playerIndex, self.loopIter = 0, 0, 0, 0
        self.playerx   = int(SCREEN_W * 0.2)
        self.playery   = int((SCREEN_H - PLAYER_H) / 2)
        self.playerx_mid = int(SCREEN_W * 0.2) + (PLAYER_W // 2)
        self.player_right = int(SCREEN_W * 0.2) + PLAYER_W
        self.playery_mid = int((SCREEN_H - PLAYER_H) // 2) + (PLAYER_H // 2)
        self.player_btm = int((SCREEN_H - PLAYER_H) // 2) + PLAYER_H
        self.basex     = 0
        self.baseShift = BASE_W - BACKGROUND_W
        self.target    = target_score
        self.pipe_gap  = self.set_mode(difficulty)  # gap between pipes

        # Randomly generate two sets of pipes
        self.pipe1, self.gapY1 = self.get_random_pipe()
        self.pipe2, self.gapY2 = self.get_random_pipe()
        self.upperPipes = [{   # upper pipe
            'x': SCREEN_W, 
            'x_mid': SCREEN_W + PIPE_W // 2,
            'x_right': SCREEN_W + PIPE_W, 
            'y': self.pipe1[0]['y']}, 
        {
            'x': SCREEN_W + (SCREEN_W // 2), 
            'x_mid': SCREEN_W + (SCREEN_W // 2) + PIPE_W // 2,
            'x_right': SCREEN_W + PIPE_W, 
            'y': self.pipe2[0]['y']}
        ]
        self.lowerPipes = [{   # upper pipe
            'x': SCREEN_W, 
            'x_mid': SCREEN_W + PIPE_W // 2,
            'x_right': SCREEN_W + PIPE_W, 
            'y': self.pipe1[1]['y']}, 
        {
            'x': SCREEN_W + (SCREEN_W // 2), 
            'x_mid': SCREEN_W + (SCREEN_W // 2) + PIPE_W // 2,
            'x_right': SCREEN_W + PIPE_W, 
            'y': self.pipe2[1]['y']}
        ]
        self.gap_loc = [self.gapY1, self.gapY2]

        for p in np.append(self.upperPipes, self.lowerPipes):
            left, right = self.get_pipe_corners(p)
            p.update({
                'corners': {
                    'left': left,
                    'right': right,
            }})

        '''---Movement and action parameters'''
        self.pipeVelX = -4
        self.playerVelY    = -9      # player's velocity along Y
        self.playerMaxVelY = 10      # max vel along Y, max descend speed
        self.playerMinVelY = -8      # min vel along Y, max ascend speed
        self.playerAccY    =  1      # players downward acceleration
        self.playerFlapAcc = -9      # players speed on flapping
        self.playerFlapped =  False  # True when player flaps
        self.msg = 'Fly, little birdie!!'

    def reset(self):
        self.__init__()
    
    def close(self):
        pygame.quit()
    
    def step(self, action):
        if self.crash: self.reset()

        pygame.event.pump()
        mod = pygame.key.get_mods()
        status = 'play'

        '''---Check for Human Input---'''
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    status = 'exit'
                if event.key == pygame.K_s and mod & pygame.KMOD_CTRL:
                    status = 'save'

        ''' Player Movement:
             action[0] == 1:  Don't Flap
             action[1] == 1:  Flap '''
        if sum(action) != 1:          # validate action
            raise ValueError('Invalid action state!')

        print('Working!')

        if action[1] == 1:
            if self.playery > -2 * PLAYER_H:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        '''---Check if player is passing through pipes--'''
        playerMidPos = self.playerx + PLAYER_W // 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_W // 2
            # Player is between the middle of pipes
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1        # player has scored
                self.scored = True

        '''---Move basex index to the left---'''
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        '''---Adjust player velocity---
        * Based on action and current position, and acceleration '''
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        #print('velY: {} | alt: {}'.format(self.playerVelY, BASE_Y - self.playery - PLAYER_H))
        self.playery += min(self.playerVelY, 
                            BASE_Y - self.playery - PLAYER_H)
        if self.playery <= 0:
            self.playery = 0

        '''---Shift pipes to the left---'''
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        
        '''---Delete pipes when they move off screen---'''
        if self.upperPipes[0]['x'] < -PIPE_W:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
            self.gap_loc.pop(0)

        
        '''---Add set of pies as first approaches left of screen---'''
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe, gapY = self.get_random_pipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
            self.gap_loc.append(gapY)
            for p in np.append(self.upperPipes, self.lowerPipes):
                left, right = self.get_pipe_corners(p)
                p.update({
                    'corners': {
                        'left': left,
                        'right': right,
                }})

        '''---Check if bird has collided with a pipe or the ground---'''
        #if reward < 0: reward = 0             # set negative reward to 0
        self.crash = self.is_crash()

        '''---Update screen to reflect state changes---'''
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], 
                       (uPipe['x'], 
                        uPipe['y'],
                        uPipe['x_mid'],
                        uPipe['y_mid'],
                        uPipe['corners']))
            SCREEN.blit(IMAGES['pipe'][1], 
                       (lPipe['x'], 
                        lPipe['y'],
                        lPipe['x_mid'],
                        lPipe['y_mid'],
                        lPipe['corners']))

        #SCREEN.blit(self.gap_loc,
        #            self.gap_loc['gapY1'],
        #            self.gap_loc['gapY2'])

        SCREEN.blit(IMAGES['base'], 
                    (self.basex, BASE_Y))

        SCREEN.blit(IMAGES['player'][self.playerIndex], 
                    (self.playerx, 
                    self.playerx_mid, 
                    self.playery,
                    self.playery_mid,
                    self.player_right,
                    self.player_btm
                    ))

        '''---Preserve frame image data to pass into neural network---'''
        observation = pygame.surfarray.array3d(pygame.display.get_surface())

        '''---Progress to the next step---'''
        pygame.display.update()
        FPSCLOCK.tick(self.fps * self.tick)  # speed up play

        for gap in self.gap_loc:
            gap['mid'] = (self.upperPipes[0]['x_mid'], gap['y_mid'])

        out = {
            'status': status,
            'scored': self.scored,
            'score': self.score,
            'terminal': self.crash,     # episode
            'target': self.target,
            'step': self.t,
            'player': {
                'x': self.playerx,
                'y': self.playery,
                'mid': (self.playerx_mid, self.playery_mid),
                'x_mid': self.playerx_mid,
                'y_mid': self.playery_mid,
                'right': self.player_right,
                'y_btm': self.player_btm,
                'y_vel': self.playerVelY,
                'flapped': self.playerFlapped,
            },
            'pipes': {
                'upper': self.upperPipes,
                'lower': self.lowerPipes,
            },
            'gaps': self.gap_loc,
        }

        reward = out
        self.t = self.t + 1
        return observation, reward, self.crash, out
    
    
    def set_mode(self, difficulty):
        '''Sets size of the gap between the upper and lower pipes'''
        mode = {'easy': 200,          # large pipe gap
                'intermediate': 150,  # medium pipe gap
                'hard': 100}          # small pipe gap
        return mode.get(difficulty)

    def get_pipe_corners(self, p):
        left = [p['x'], p['y']]
        right = [p['x_right'], p['y']]
        return left, right

    def get_random_pipe(self):
        '''returns a randomly generated pipe'''
        # y of gap between upper and lower pipe
        gapY = rand.randrange(0, 
            int(BASE_Y * 0.6 - self.pipe_gap))
        gapY += int(BASE_Y * 0.2)
        pipeX = SCREEN_W + 10

        pipe_n = [{   # upper pipe
            'x': pipeX, 'x_mid': pipeX + PIPE_W // 2,
            'x_right': pipeX + PIPE_W, 'y': gapY - PIPE_H},       
        {           # lower pipe
            'x': pipeX, 'x_mid': pipeX + PIPE_W // 2,
            'x_right': pipeX + PIPE_W, 'y': gapY + self.pipe_gap}]

        gap_n = { 
            'y': gapY,   # gap pos for pipe set 1
            'y_mid': gapY + self.pipe_gap // 2,
            'y_btm': gapY + self.pipe_gap }

        return pipe_n, gap_n

    def draw_polygon(self):

    
    
    def is_crash(self):
        '''returns True if player collides with base or pipes'''
        if self.playery + PLAYER_H >= BASE_Y - 1:
            return True           # Player Crashed into ground
        else:
            playerRect = pygame.Rect(self.playerx, self.playery,
                                     PLAYER_W, PLAYER_H)

            for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
                # upper and lower pipe boundary
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], 
                                        PIPE_W, PIPE_H)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], 
                                        PIPE_W, PIPE_H)
                # Masks for collision bounding boxes
                pHitMask = HITMASKS['player'][self.playerIndex]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixel_collision(playerRect, uPipeRect, 
                                                pHitMask, uHitmask)
                lCollide = self.pixel_collision(playerRect, lPipeRect, 
                                                pHitMask, lHitmask)

                if uCollide or lCollide:  # Player Crashed into pipe
                    return True
        # No Crash
        return False

    def pixel_collision(self, rect1, rect2, hitmask1, hitmask2):
        '''Checks if two objects collide'''
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False
