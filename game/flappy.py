
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
sys.path.append('../')
sys.path.append('../learner')
sys.path.append('/learner')
import pygame
import pygame.locals
import pygame.surfarray as surfarray
import numpy as np
import random as rand
import pprint as pp
from pygame.locals import *
from itertools import cycle
import game.flappy_load as fl
import collections


pygame.init()
# Graphic Settings
SCREEN_W = 288
SCREEN_H = 512
CLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption('Flappy Bird')
pygame.display.set_icon(pygame.image.load('game/flappy.ico'))
IMAGES, HITMASKS = fl.load()

class Environment:    
    def __init__(self, user_setting, util, **kwargs): 
        '''---Initialize New Game---'''
        if util != None: 
            self.util = util

        self.u_s = user_setting

        self.settings = {
            'screen': {
                'h': SCREEN_H,
                'w': SCREEN_W,
                'background_w': IMAGES['background'].get_width(),
                'base_x': 0,
                'base_y': SCREEN_H * 0.79,
                'base_w': IMAGES['base'].get_width(),
                'base_sft': IMAGES['base'].get_width() - \
                            IMAGES['background'].get_width(),
            },
            'pygame': {
                'clock': CLOCK,
                'target': user_setting['target'],
                'fps': user_setting['fps'],
                'tick': user_setting['tick'],
                'difficulty': user_setting['difficulty'],
                'name': user_setting['name'],
            },
            'player': {
                'x': int(SCREEN_W * 0.2),
                'y': int((SCREEN_H - IMAGES['player'][0].get_height()) // 2),
                'h': IMAGES['player'][0].get_height(),
                'w': IMAGES['player'][0].get_width(),
                'idx_gen': cycle([0, 1, 2, 1]),
                'idx': 0,
                'y_vel':-9,        # player's velocity along Y
                'y_vel_max': 10,   # max vel along Y, max descend speed
                'y_vel_min': -8,   # min vel along Y, max ascend speed
                'y_acc': 1,        # players downward acceleration
                'rot': 45,         # player's rotation
                'rot_vel': 3,      # angular speed
                'rot_thr': 20,     # rotation threshold
                'flap_acc': -9,    # players speed on flapping
                'flapped': False,  # True when player flaps
            },
            'pipe': {
                'x_vel': -4,
                'h': IMAGES['pipe'][0].get_height(),
                'w': IMAGES['pipe'][0].get_width(),
                'gap': {
                    'size': self.set_mode(user_setting['difficulty']),
                    'loc': [],
                },
                'upper': [],
                'lower': [],

            },
            'track': {
                'scored': False,
                'score': 0,
                'status': 'play',
                'crash': False,
                'loopIter': 0,
        }}

        self.screen = self.settings['screen']
        self.player = self.settings['player']
        self.track = self.settings['track']
        self.pipe = self.settings['pipe']
        self.gap = self.pipe['gap']
        
        self.player.update({
            'x_mid': self.player['x'] + (self.player['w'] // 2),
            'x_right': self.player['x'] + self.player['w'],
            'y_mid': self.player['y'] + (self.player['h'] // 2),
            'y_btm': self.player['y'] + self.player['h'], 
            'vis_rot': self.player['rot_thr'],
        })

        # Randomly generate two sets of pipes
        pipe1, gapY1 = self.get_random_pipe(self.pipe)
        pipe2, gapY2 = self.get_random_pipe(self.pipe)
        self.pipe['upper'] = [pipe1[0], pipe2[0]]
        self.pipe['lower'] = [pipe1[1], pipe2[1]]
        self.pipe['gap']['loc'] = [gapY1, gapY2]

        for pipe in np.append(self.pipe['upper'], self.pipe['lower']):
            left, right = self.get_pipe_corners(pipe)
            pipe.update({
                'corners': {
                    'left': left,
                    'right': right,
            }})


    def get_pipe_corners(self, pipe):
        left = [pipe['x'], pipe['y']]
        right = [pipe['x_right'], pipe['y']]
        return left, right
    
    def get_random_pipe(self, pipe):
        '''returns a randomly generated pipe'''
        # y of gap between upper and lower pipe
        gapY = rand.randrange(0, 
            int(self.screen['base_y'] * 0.6 - pipe['gap']['size']))
        gapY += int(self.screen['base_y'] * 0.2)
        pipeX = self.screen['w'] + 10

        pipe_n = [{   # upper pipe
            'x': pipeX, 'x_mid': pipeX + (pipe['w'] // 2),
            'x_right': pipeX + pipe['w'], 'y': gapY - pipe['h']},       
        {           # lower pipe
            'x': pipeX, 'x_mid': pipeX + (pipe['w'] // 2),
            'x_right': pipeX + pipe['w'], 'y': gapY + pipe['gap']['size']}]
        gap = [{ 
            'top': gapY,   # gap pos for pipe set 1
            'mid': gapY + (pipe['gap']['size'] // 2),
            'btm': gapY + pipe['gap']['size'] }]

        return pipe_n, gap

    def quit_game(self):
        pygame.quit()

    def step(self, action):
        if self.track['crash']:    # bird crashed!
            self.__init__(user_setting=self.u_s,
                          util=self.util)

        pygame.event.pump()
        mod = pygame.key.get_mods()
        self.track['status'] = 'play'
        self.track['scored'] = False

        '''---Check for Human Input---'''
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.track['status'] = 'exit'
                if event.key == pygame.K_s and mod & pygame.KMOD_CTRL:
                    self.track['status'] = 'save'

        ' Rotate the player'
        if self.player['rot'] > -90:
            self.player['rot'] -= self.player['rot_vel']

        ''' Player Movement:
             action[0] == 1:  Don't Flap
             action[1] == 1:  Flap '''
        if sum(action) != 1:          # validate action
            raise ValueError('Invalid action state!')

        if action[1] == 1:
            if self.player['y']> -2 * self.player['h']:
                self.player['y_vel'] = self.player['flap_acc']
                self.player['flapped'] = True

        '''---Check if player is passing through pipes--'''
        self.player['x_mid'] = self.player['x'] + self.player['w'] // 2
        for p in self.pipe['upper']:
            p['x_mid'] = p['x'] + self.pipe['w'] // 2
            # Player is between the middle of pipes
            if p['x_mid'] <= self.player['x_mid'] < p['x_mid'] + 4:
                self.track['score'] += 1
                self.track['scored'] = True

        '''---Move basex index to the left---'''
        if (self.track['loopIter'] + 1) % 3 == 0:
            self.player['idx'] = next(self.player['idx_gen'])
        self.track['loopIter'] = (self.track['loopIter'] + 1) % 30
        self.screen['base_x']= -((-self.screen['base_x'] + 100) % self.screen['base_sft'])

        '''---Adjust player velocity---
        * Based on action and current position, and acceleration '''
        if self.player['y_vel'] < self.player['y_vel_max'] and not self.player['flapped']:
            self.player['y_vel'] = self.player['y_vel'] + self.player['y_acc']
        if self.player['flapped']: 
            self.player['flapped'] = False
            self.playerRot = 45
        #print('velY: {} | alt: {}'.format(self.playerVelY, BASE_Y - self.playery - PLAYER_H))
        self.player['y'] = self.player['y'] + min(self.player['y_vel'],
                      self.screen['base_y'] - self.player['y'] - self.player['h'])
        if self.player['y'] <= 0: self.player['y'] = 0

        '''---Shift pipes to the left---'''
        for u, l in zip(self.pipe['upper'], self.pipe['lower']):
            u['x'] += self.pipe['x_vel']
            l['x'] += self.pipe['x_vel']

        '''---Add set of pipes as first approaches left of screen---'''
        if 0 < self.pipe['upper'][0]['x'] < 5:
            pipe_n, gapY = self.get_random_pipe(self.pipe)
            self.pipe['upper'].append(pipe_n[0])
            self.pipe['lower'].append(pipe_n[1])
            self.gap['loc'].append(gapY)

        '''---Delete pipes when they move off screen---'''
        if self.pipe['upper'][0]['x'] < -self.pipe['w']:
            self.pipe['upper'].pop(0)
            self.pipe['lower'].pop(0)
            self.gap['loc'].pop(0)

        # Player rotation has a limit
        self.player['vis_rot'] = self.player['rot_thr']
        if self.player['rot'] <= self.player['rot_thr']:
            self.player['vis_rot'] = self.player['rot']

        '''---Check if bird has collided with a pipe or the ground---'''
        crash = self.is_crash()  # set as last frame
        if crash: self.track['crash'] = True

        '''---Update screen to reflect state changes---'''
        for u, l in zip(self.pipe['upper'], self.pipe['lower']):
            SCREEN.blit(IMAGES['pipe'][0], (u['x'], l['y']))
            SCREEN.blit(IMAGES['pipe'][1], (u['x'], l['y']))

        SCREEN.blit(IMAGES['base'], (self.screen['base_x'], 
                                     self.screen['base_y']))

        SCREEN.blit(IMAGES['player'][self.player['idx']], 
                    (self.player['x'], self.player['y']))

        '''---Preserve frame image data to pass into neural network---'''
        scn_cap = pygame.surfarray.array3d(pygame.display.get_surface())

        '''---Progress to the next step---'''
        pygame.display.update()
        CLOCK.tick(self.u_s['fps'] * self.u_s['tick'])  # speed up play
        update = {
            'player': self.player,
            'pipe': self.pipe,
            'track': self.track,
        }

        return scn_cap, update

    def set_mode(self, difficulty):
        '''Sets size of the gap between the upper and lower pipes'''
        mode = {'easy': 200,          # large pipe gap
                'intermediate': 150,  # medium pipe gap
                'hard': 100}          # small pipe gap
        return mode.get(difficulty)

    def is_crash(self):
        '''returns True if player collides with base or pipes'''
        if self.player['y'] + self.pipe['h'] >= self.screen['base_y'] - 1:
            return True           # Player Crashed into ground
        else:
            playerRect = pygame.Rect(self.player['x'], self.player['y'],
                                     self.pipe['w'], self.pipe['h'])
            for u, l in zip(self.pipe['upper'], self.pipe['lower']):
                # upper and lower pipe boundary
                uRect = pygame.Rect(u['x'], u['y'],
                                    self.pipe['w'], self.pipe['h'])
                lRect = pygame.Rect(l['x'], l['y'],
                                    self.pipe['w'], self.pipe['h'])
                # Masks for collision bounding boxes
                pHitMask = HITMASKS['player'][self.player['idx']]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCol = self.pixel_collision(
                    playerRect, uRect, pHitMask, uHitmask)
                lCol = self.pixel_collision(
                    playerRect, lRect, pHitMask, lHitmask)
                # rotate player on pipe crash
                if uCol or lCol: 
                    self.player['rot'] = -self.player['rot_vel']
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
