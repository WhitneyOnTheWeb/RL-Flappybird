
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
import flappy_load
import random
import pygame
import pygame.surfarray as surfarray
import numpy as np
from pygame.locals import *
from itertools import cycle

# Graphic Settings
pygame.init()
pygame.display.set_caption('Flappy Bird')
SCREEN_W  = 288
SCREEN_H  = 512
FPS       = 30
FPSCLOCK  = pygame.time.Clock()
SCREEN    = pygame.display.set_mode((SCREEN_W, SCREEN_H))

# Obstacle Settings
PIPE_GAP  = 100 # gap between upper and lower pipes
BASE_Y    = SCREEN_H * 0.79 # amount base can shift left

# Game Assets
IMAGES, HITMASKS = flappy_load.load()

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

class GameState():
    def __init__(self):
        self.score = self.playerIndex = self.loopiter = 0
        self.playerx   = int(SCREEN_W * 0.2)
        self.playery   = int((SCREEN_H - PLAYER_H) / 2)
        self.basex     = 0
        self.baseShift = BASE_W - BACKGROUND_W

        pipe1 = getRandomPipe()
        pipe2 = getRandomPipe()
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

        # player velocity, max velocity, acceleration, rotation
        self.pipeVelX = -4
        self.playerVelY    =   0    # player's velocity along Y
        self.playerMaxVelY =  10    # max vel along Y, max descend speed
        self.playerMinVelY =  -8    # min vel along Y, max ascend speed
        self.playerAccY    =   1    # players downward acceleration
        self.playerRot     =  45    # player's rotation
        self.playerVelRot  =   3    # angular speed
        self.playerRotThr  =  20    # rotation threshold
        self.playerFlapAcc =  -9    # players speed on flapping
        self.playerFlapped =  False # True when player flaps

    def step(self, action):
        pygame.event.pump()

        # Reward continued play
        reward = 0.1
        terminal = False

        # Player Movement----------------------------------------------------------
        # action[0] == 1:  Don't Flap
        # action[1] == 1:  Flap
        if sum(action) != 1:
            raise ValueError('More than 1 action!')

        if action[1]:
            if self.playery > -2 * PLAYER_H:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # Check if player is passing through pipes
        playerMidPos = self.playerx + PLAYER_W / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_W / 2
            # Player is between the middle of pipes
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                reward += 1  # add large reward for scoring

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # track player movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
            self.playerRot = 45 # rotate to cover the threshold
        self.playery += min(self.playerVelY, 
                            BASE_Y - self.playery - PLAYER_H)
        if self.playery < 0:
            self.playery = 0

        # Pipe Movement------------------------------------------------------------
        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its off the screen
        if self.upperPipes[0]['x'] < -PIPE_W:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # Check for Collision------------------------------------------------------
        crash = checkCrash({'x': self.playerx,
                            'y': self.playery,
                            'index': self.playerIndex},
                        self.upperPipes, 
                        self.lowerPipes)
        if crash:
            terminal = True  # set as last frame 
            self.__init__()
            reward -= 5      # very large penalty if crash occurs
        
        # Update Screen------------------------------------------------------------
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], 
                        (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], 
                        (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], 
                    (self.basex, BASE_Y))

        # limit player rotation
        self.visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            self.visibleRot = self.playerRot

        SCREEN.blit(IMAGES['player'][self.playerIndex], 
                    (self.playerx, self.playery))

        # preserve frame as image to be passed into Deep-Q CNN
        image = pygame.surfarray.array3d(pygame.display.get_surface())

        # Move to Next Frame-------------------------------------------------------
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        
        return image, reward, score, terminal

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY  = random.randrange(0, int(BASE_Y * 0.6 - PIPE_GAP))
        gapY += int(BASE_Y * 0.2)
        pipeX = SCREEN_W + 10

        return [
            {'x': pipeX, 'y': gapY - PIPE_H},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPE_GAP} # lower pipe
        ]

    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth  = 0 # width of numbers to be shown
        for digit in scoreDigits:
            totalWidth += IMAGES['numbers'][digit].get_width()

        Xoffset = (SCREEN_W - totalWidth) / 2
        for digit in scoreDigits:
            SCREEN.blit(IMAGES['numbers'][digit], 
                        (Xoffset, SCREEN_H * 0.1))
            Xoffset += IMAGES['numbers'][digit].get_width()

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collides with base or pipes."""
        pi = player['index']

        if player['y'] + PLAYER_H >= BASE_Y - 1:  # Player Crashed
            return True
        else:
            playerRect = pygame.Rect(player['x'], player['y'],
                        PLAYER_W, PLAYER_H)

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], 
                                        PIPE_W, PIPE_H)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], 
                                        PIPE_W, PIPE_H)

                # player and upper/lower pipe hitmasks
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
        """Checks if two objects collide and not just their rects"""
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
