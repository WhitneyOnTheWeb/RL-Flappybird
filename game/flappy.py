
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
import pygame.surfarray as surfarray
import numpy as np
import random as rand
from pygame.locals import *
from itertools import cycle
import game.flappy_load as fl

# Graphic Settings
SCREEN_W  = 288
SCREEN_H  = 512
FPS       = 30
FPSCLOCK  = pygame.time.Clock()
SCREEN    = pygame.display.set_mode((SCREEN_W, SCREEN_H))

# Obstacle Settings
BASE_Y    = SCREEN_H * 0.79 # amount base can shift left

# Game Assets
IMAGES, HITMASKS = fl.load()

# Calculate Asset Sizes
PLAYER_W     = IMAGES['player'][0].get_width()
PLAYER_H     = IMAGES['player'][0].get_height()
PIPE_W       = IMAGES['pipe'][0].get_width()
PIPE_H       = IMAGES['pipe'][0].get_height()
BACKGROUND_W = IMAGES['background'].get_width()
BACKGROUND_H = IMAGES['background'].get_height()
BASE_W       = IMAGES['base'].get_width()

# iterator used to change playerIndex after every 5th iteration
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

class GameState:
    def __init__(self, target_score = 40, difficulty = 'hard'):
        '''---Initialize New Game---'''
        pygame.init()
        pygame.display.set_caption('Flappy Bird')
        
        self.score = self.reward = self.playerIndex = self.loopIter = 0
        self.playerx   = int(SCREEN_W * 0.2)
        self.playery   = int((SCREEN_H - PLAYER_H) / 2)
        self.basex     = 0
        self.baseShift = BASE_W - BACKGROUND_W
        self.target    = target_score
        self.pipe_gap  = self.set_mode(difficulty)  # gap between pipes
        self.terminal  = False               # sets game as in progress

        pipe1 = self.getRandomPipe()
        pipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': SCREEN_W, 
             'y': pipe1[0]['y']},
            {'x': SCREEN_W + (SCREEN_W / 2), 
             'y': pipe2[0]['y']}]
        self.lowerPipes = [
            {'x': SCREEN_W, 
             'y': pipe1[1]['y']},
            {'x': SCREEN_W + (SCREEN_W / 2), 
             'y': pipe2[1]['y']}]

        '''---Movement and action parameters'''
        self.pipeVelX = -4
        self.playerVelY    =   0      # player's velocity along Y
        self.playerMaxVelY =  10      # max vel along Y, max descend speed
        self.playerMinVelY =  -8      # min vel along Y, max ascend speed
        self.playerAccY    =   1      # players downward acceleration
        self.playerRot     =  45      # player's rotation
        self.playerVelRot  =   3      # angular speed
        self.playerRotThr  =  20      # rotation threshold
        self.playerFlapAcc =  -9      # players speed on flapping
        self.playerFlapped =  False   # True when player flaps

    def step(self, action):
        '''---If game over, reset game state---'''
        if self.terminal: self.__init__() 
        pygame.event.pump()

        ''' Player Movement:
             action[0] == 1:  Don't Flap
             action[1] == 1:  Flap '''
        if sum(action) != 1:          # validate action
            raise ValueError('Invalid action state!')

        if action[1]:
            if self.playery > -2 * PLAYER_H:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        '''---Check if player is passing through pipes--'''
        playerMidPos = self.playerx + PLAYER_W / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_W / 2
            # Player is between the middle of pipes
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1      # player has scored
                self.reward += 1     # large reward for scoring

                '''---Increase reward as score approaches target score---'''
                if self.score >= self.target:           self.reward += 8
                elif self.score >= self.target * 0.75:  self.reward += 5
                elif self.score >= self.target * 0.5:   self.reward += 3
                elif self.score >= self.target * 0.25:  self.reward += 2

        '''---Shift player index forward---'''
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        '''---Rotate player based on current rotation and velocity---'''
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        '''---Adjust player velocity---
        * Based action and current position, rotation, and acceleration '''
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
            self.playerRot = 45               # rotate to cover the threshold
        self.playery += min(self.playerVelY, 
                            BASE_Y - self.playery - PLAYER_H)
        if self.playery < 0:
            self.playery = 0

        '''---Shift pipes to the left---'''
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        '''---Add second pipe as first pipe reaches left side of screen---'''
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        '''---Delete pipes when they move off screen---'''
        if self.upperPipes[0]['x'] < -PIPE_W:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        '''---Check if bird has collided with a pipe or the ground---'''
        crash = self.checkCrash({'x': self.playerx,
                                 'y': self.playery,
                                 'index': self.playerIndex},
                                self.upperPipes, 
                                self.lowerPipes)
        if crash:
            self.terminal  = True   # set as last frame 
            self.reward -= 5      # very large penalty if crash occurs
        else: self.reward += 0.01    # small reward for continued flying
        
        '''---Update screen to reflect state changes---'''
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], 
                        (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], 
                        (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], 
                    (self.basex, BASE_Y))

        self.visibleRot = self.playerRotThr      # limit player rotation
        if self.playerRot <= self.playerRotThr:
            self.visibleRot = self.playerRot

        SCREEN.blit(IMAGES['player'][self.playerIndex], 
                    (self.playerx, self.playery))

        '''---Preserve frame image data to pass into neural network---'''
        self.frame = pygame.surfarray.array3d(pygame.display.get_surface())

        '''---Progress to the next step---'''
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        
        return self.frame, self.reward, self.score, self.terminal

    def set_mode(self, difficulty):
        '''Sets size of the gap between the upper and lower pipes'''
        mode = {'easy': 200,          # large pipe gap
                'intermediate': 150,  # medium pipe gap
                'hard': 100}          # small pipe gap
        return mode.get(difficulty)

    def getRandomPipe(self):
        '''returns a randomly generated pipe'''
        # y of gap between upper and lower pipe
        gapY  = rand.randrange(0, int(BASE_Y * 0.6 - self.pipe_gap))
        gapY += int(BASE_Y * 0.2)
        pipeX = SCREEN_W + 10

        return [
            {'x': pipeX, 'y': gapY - PIPE_H},       # upper pipe
            {'x': pipeX, 'y': gapY + self.pipe_gap} # lower pipe
        ]

    def checkCrash(self, player, upperPipes, lowerPipes):
        '''returns True if player collides with base or pipes'''
        pi = player['index']

        if player['y'] + PLAYER_H >= BASE_Y - 1:
            return True           # Player Crashed into Ground
        else:
            playerRect = pygame.Rect(player['x'], player['y'],
                        PLAYER_W, PLAYER_H)

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe boundary
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], 
                                        PIPE_W, PIPE_H)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], 
                                        PIPE_W, PIPE_H)

                # Masks for collision bounding boxes
                pHitMask = HITMASKS['player'][pi]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, 
                                               pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, 
                                               pHitMask, lHitmask)

                if uCollide or lCollide:  # Player Crashed
                    return True
        # No Crash
        return False

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
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

    def showScore(self):
        '''displays score in center of screen'''
        scoreDigits = [int(x) for x in list(str(self.score))]
        totalWidth  = 0 # width of numbers to be shown
        for digit in scoreDigits:
            totalWidth += IMAGES['numbers'][digit].get_width()

        Xoffset = (SCREEN_W - totalWidth) / 2
        for digit in scoreDigits:
            SCREEN.blit(IMAGES['numbers'][digit], 
                        (Xoffset, SCREEN_H * 0.1))
            Xoffset += IMAGES['numbers'][digit].get_width()
