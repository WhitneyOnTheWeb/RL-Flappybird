
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
import random as rand
from pygame.locals import *
from itertools import cycle
import game.flappy_load as fl

# Graphic Settings
SCREEN_W = 288
SCREEN_H = 512
FPSTICK  = 4

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

class GameState:
    def __init__(self, target_score = 40, difficulty = 'hard', fps = 30):
        

        '''---Initialize New Game---'''        
        self.name = 'FlappyBird'
        self.icon = pygame.image.load('game/flappy.ico')
        self.width = SCREEN_W
        self.height = SCREEN_H
        self.fps = fps
        self.mode = difficulty

        pygame.display.set_caption(self.name)
        pygame.display.set_icon(self.icon)

        self.terminal = False
        self.score = self.reward = self.playerIndex = self.loopIter = 0
        self.playerx   = int(SCREEN_W * 0.2)
        self.playery   = int((SCREEN_H - PLAYER_H) / 2)
        self.basex     = 0
        self.baseShift = BASE_W - BACKGROUND_W
        self.target    = target_score
        self.pipe_gap  = self.set_mode(difficulty)  # gap between pipes

        # Randomly generate two sets of pipes
        self.pipe1, self.gapY1 = self.get_random_pipe()
        self.pipe2, self.gapY2 = self.get_random_pipe()
        self.upperPipes = [
            {'x': SCREEN_W, 
             'y': self.pipe1[0]['y']},
            {'x': SCREEN_W + (SCREEN_W / 2), 
             'y': self.pipe2[0]['y']}]
        self.lowerPipes = [
            {'x': SCREEN_W, 
             'y': self.pipe1[1]['y']},
            {'x': SCREEN_W + (SCREEN_W / 2), 
             'y': self.pipe2[1]['y']}]
        self.gapPos = [
            {'top' : self.gapY1,   # gap pos for pipe set 1
             'btm' : self.gapY1 + self.pipe_gap},
            {'top' : self.gapY2,   # gap pos for pipe set 1
             'btm' : self.gapY2 + self.pipe_gap},
        ]

        '''---Movement and action parameters'''
        '''!!!Updated physics parameters to limit rapid ascent when slowed to 30 FPS!!!'''
        self.pipeVelX = -4
        self.playerVelY    =   0      # player's velocity along Y
        self.playerMaxVelY =   5      # max vel along Y, max descend speed
        self.playerMinVelY =  -4      # min vel along Y, max ascend speed
        self.playerAccY    =   1      # players downward acceleration
        self.playerFlapAcc =  -5      # players speed on flapping
        self.playerFlapped =  False   # True when player flaps

    def quit_game(self):
        pygame.quit()
    
    def step(self, action):
        if self.terminal: self.__init__() 

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

        reward = 1 #reward getting another frame

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
                reward += 200     # large reward for scoring

                '''---Increase reward as score approaches target score---'''
                if self.score >= self.target:           reward += 1000
                elif self.score >= self.target * 0.75:  reward += 600
                elif self.score >= self.target * 0.5:   reward += 400
                elif self.score >= self.target * 0.25:  reward += 300

        if playerMidPos >= self.upperPipes[0]['x'] + PIPE_W // 2:
            gap = self.gapPos[0]
        else: gap = self.gapPos[1]           # gap to check against
        # calculate reward based on distance above / below pipe gap
        if gap['btm'] > self.playery > gap['top']:
            reward += 5
            msg = 'Stay here!'                             # reward
        elif self.playery <= gap['top'] + 5:   # penalize
            msg = 'Go down!'
            reward -= (gap['top'] - self.playery) *.01
        elif self.playery >= gap['btm'] - 5:
            msg = 'Go up!'
            reward -= (self.playery - gap['btm']) *.01
        else: msg = 'Almost right!'                        # no reward

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

        '''---Add set of pies as first approaches left of screen---'''
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe, gapY = self.get_random_pipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        '''---Delete pipes when they move off screen---'''
        if self.upperPipes[0]['x'] < -PIPE_W:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        '''---Check if bird has collided with a pipe or the ground---'''
        crash = self.is_crash()
        if crash:
            self.terminal  = True    # set as last frame 
            if self.playery <= gap['top']: 
                penalty = (gap['top'] - self.playery) * .1
            elif self.playery >= gap['btm']:
                penalty = (self.playery - gap['btm']) * .1
            else: penalty = 1
            reward -= (1 + penalty)   # penalty
            msg = 'Boom!'

        '''Scale and constrain reward values, add to episode reward'''
        if reward < 0: reward = 0      # set negative reward to 0
        reward = np.tanh(reward)       # reward between [-1, 1]
        self.reward += reward

        '''---Update screen to reflect state changes---'''
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], 
                       (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], 
                       (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], 
                    (self.basex, BASE_Y))

        SCREEN.blit(IMAGES['player'][self.playerIndex], 
                    (self.playerx, self.playery))

        '''---Preserve frame image data to pass into neural network---'''
        self.frame = pygame.surfarray.array3d(pygame.display.get_surface())

        '''---Progress to the next step---'''
        pygame.display.update()
        FPSCLOCK.tick(self.fps * FPSTICK)  # speed up play
        
        return self.frame, reward, self.reward, self.score, self.terminal, msg, status

    def set_mode(self, difficulty):
        '''Sets size of the gap between the upper and lower pipes'''
        mode = {'easy': 200,          # large pipe gap
                'intermediate': 150,  # medium pipe gap
                'hard': 100}          # small pipe gap
        return mode.get(difficulty)

    def get_random_pipe(self):
        '''returns a randomly generated pipe'''
        # y of gap between upper and lower pipe
        gapY  = rand.randrange(0, int(BASE_Y * 0.6 - self.pipe_gap))
        gapY += int(BASE_Y * 0.2)
        pipeX = SCREEN_W + 10

        return [
            {'x': pipeX, 'y': gapY - PIPE_H},       # upper pipe
            {'x': pipeX, 'y': gapY + self.pipe_gap} # lower pipe
        ], gapY
    
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
