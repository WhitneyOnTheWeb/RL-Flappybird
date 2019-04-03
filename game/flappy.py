
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
IMAGES, HITMASKS = fl.load()

class Environment:    
    def __init__(self, params=None, **kwargs): 
        pygame.init()
        game = params['game']

        '''---Initialize New Game---'''
        game['settings'].update({
            'screen': {
                'icon': pygame.image.load('game/flappy.ico'),
                'display': pygame.display.set_mode((SCREEN_W, SCREEN_H)),
            },
            'pygame': {'clock': pygame.time.Clock()},
            'player': {'idx_gen': cycle([0, 1, 2, 1])},
            'pipe': {
                'gap': {
                    'size': self.set_mode(game['difficulty']),
                    'loc': {},
                },
                'upper': {},
                'lower': {},
        }})

        settings = game['settings']
        pipe = settings['pipe']
        screen = settings['screen']

        pygame.display.set_caption(settings['name'])
        pygame.display.set_icon(screen['icon'])

        # Randomly generate two sets of pipes
        pipe1, gapY1 = self.get_random_pipe(pipe, screen)
        pipe2, gapY2 = self.get_random_pipe(pipe, screen)
        game['settings']['pipe']['upper'].update([ 
          { 'x': screen['w'],
            'x_mid': screen['w'] + (pipe['w'] // 2),
            'x_right': screen['w'] + pipe['w'], 
            'y': pipe1[0]['y'] },
          { 'x': screen['w'] + (screen['w'] // 2),
            'x_mid': (screen['w'] + (screen['w'] // 2)) + \
                (pipe['w'] // 2),
            'x_right': (screen['w'] + (screen['w'] // 2)) + \
                pipe['w'], 
            'y': pipe2[0]['y'] }])

        game['settings']['pipe']['lower'].update([
            { 'x': screen['w'],
              'x_mid': screen['w'] + (pipe['w'] // 2),
              'x_right': screen['w'] + pipe['w'], 
              'y': pipe1[1]['y'] },
            { 'x': screen['w'] + (screen['w'] // 2),
              'x_mid': (screen['w'] + (screen['w'] // 2)) + \
                  (pipe['w'] // 2),
              'x_right': (screen['w'] + (screen['w'] // 2)) + \
                  pipe['w'], 
              'y': pipe2[1]['y'] }])

        game['settings']['pipe']['gap']['loc'].update([
            { 'top': gapY1,   # gap pos for pipe set 1
              'mid': gapY1 + (pipe['gap']['size'] // 2),
              'btm': gapY1 + pipe['gap']['size'] },
            { 'top': gapY2,   # gap pos for pipe set 1
              'mid': gapY2 + (pipe['gap']['size'] // 2),
              'btm': gapY2 + pipe['gap']['size'] }])

        for pipe in np.append(pipe['upper'], pipe['lower']):
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
    
    def quit_game(self):
        pygame.quit()

    def step(self, action, params):
        settings = params['game']['settings']
        env = settings['pygame']
        screen = settings['screen']
        track = settings['track']
        player = settings['player']
        pipe = settings['pipe']
        gap = pipe['gap']
        t = params['session']['episode']['step']['t']

        if track['crash']:    # bird crashed!
            self.__init__()

        pygame.event.pump()
        mod = pygame.key.get_mods()
        track.update({
            'status': 'play',
            'scored': False,
        })

        '''---Check for Human Input---'''
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    track['status'] = 'exit'
                if event.key == pygame.K_s and mod & pygame.KMOD_CTRL:
                    track['status'] = 'save'

        ' Rotate the player'
        if player['rot'] > -90:
            player['rot'] -= player['rot_vel']

        ''' Player Movement:
             action[0] == 1:  Don't Flap
             action[1] == 1:  Flap '''
        if sum(action) != 1:          # validate action
            raise ValueError('Invalid action state!')

        if action[1] == 1:
            if player['y']> -2 * player['h']:
                player['y_vel'] = player['flap_acc']
                player['flapped'] = True

        '''---Check if player is passing through pipes--'''
        player['x_mid'].update(player['x'] + player['w'] // 2)
        for pipe in pipe['upper']:
            pipe['x_mid'].update(pipe['x'] + pipe['w'] // 2)
            # Player is between the middle of pipes
            if pipe['x_mid'] <= player['x_mid'] < pipe['x_mid'] + 4:
                track.update({
                    'score': track['score'] + 1,
                    'scored': True,
                })

        '''---Move basex index to the left---'''
        if (track['loopIter'] + 1) % 3 == 0:
            player['idx'] = next(player['idx_gen'])
        track['loopIter'].update((track['loopIter'] + 1) % 30)
        screen['base_x'].update(
            -((-screen['base_x'] + 100) % screen['base_sft']))

        '''---Adjust player velocity---
        * Based on action and current position, and acceleration '''
        if player['y_vel'] < player['y_vel_max'] and not player['flapped']:
            player['y_vel'].update(player['y_vel'] + player['y_acc'])
        if player['flapped']:
            player['flapped'] = False
            # rotate to cover threshold (calculated in visible rotation)
            self.playerRot = 45
        #print('velY: {} | alt: {}'.format(self.playerVelY, BASE_Y - self.playery - PLAYER_H))
        player['y'].update(player['y'] + min(player['y_vel'],
                            screen['base_y'] - player['y'] - player['h']))
        if player['y'] <= 0: player['y'].update(0)

        '''---Shift pipes to the left---'''
        for u, l in zip(pipe['upper'], pipe['lower']):
            u['x'] += pipe['x_vel']
            l['x'] += pipe['x_vel']

        '''---Add set of pipes as first approaches left of screen---'''
        if 0 < pipe['upper'][0]['x'] < 5:
            pipe_n, gapY = self.get_random_pipe(pipe, screen)
            pipe['upper'].append(pipe_n[0])
            pipe['lower'].append(pipe_n[1])
            gap['loc'].append(gapY)

        '''---Delete pipes when they move off screen---'''
        if pipe['upper'][0]['x'] < -pipe['w']:
            pipe['upper'].pop(0)
            pipe['lower'].pop(0)
            gap['loc'].pop(0)

        # Player rotation has a limit
        player['vis_rot'].update(player['rot_thr'])
        if player['rot'] <= player['rot_thr']:
            player['vis_rot'].update(player['rot'])

        '''---Check if bird has collided with a pipe or the ground---'''
        crash = self.is_crash(player, pipe, screen, env)  # set as last frame
        if crash: track['crash'].update(True)    

        '''---Update screen to reflect state changes---'''
        for u, l in zip(pipe['upper'], pipe['lower']):
            screen['display'].blit(env['images']['pipe'][0],
                        (u['x'], l['y']))
            screen['display'].blit(env['images']['pipe'][1],
                        (u['x'], l['y']))

        screen['display'].blit(env['images']['base'],
                    (screen['base_x'], screen['base_y']))

        screen['display'].blit(env['images'][player['idx']],
                    (player['x'], player['y']))

        '''---Preserve frame image data to pass into neural network---'''
        scn_cap = pygame.surfarray.array3d(pygame.display.get_surface())

        '''---Progress to the next step---'''
        pygame.display.update()
        env['clock'].tick(env['fps'] * env['tick'])  # speed up play
        params['game'].update({
            'screen': {screen},
            'pygame': {env},
            'player': {player},
            'pipe': {pipe},
            'track': {track},
        })
        return scn_cap, track['crash'], track['status']

    def set_mode(self, difficulty):
        '''Sets size of the gap between the upper and lower pipes'''
        mode = {'easy': 200,          # large pipe gap
                'intermediate': 150,  # medium pipe gap
                'hard': 100}          # small pipe gap
        return mode.get(difficulty)

    def get_random_pipe(self, pipe, screen):
        '''returns a randomly generated pipe'''
        # y of gap between upper and lower pipe
        gapY = rand.randrange(0, 
            int(screen['base_y'] * 0.6 - pipe['gap']['size']))
        gapY += int(screen['base_y'] * 0.2)
        pipeX = screen['w'] + 10

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

    def is_crash(self, player, pipe, screen, env):
        '''returns True if player collides with base or pipes'''
        if player['y'] + pipe['h'] >= screen['base_y'] - 1:
            return True           # Player Crashed into ground
        else:
            playerRect = pygame.Rect(player['x'], player['y'],
                                     pipe['w'], pipe['h'])
            for u, l in zip(pipe['upper'], pipe['lower']):
                # upper and lower pipe boundary
                uRect = pygame.Rect(u['x'], u['y'],
                                        pipe['w'], pipe['h'])
                lRect = pygame.Rect(l['x'], l['y'],
                                        pipe['w'], pipe['h'])
                # Masks for collision bounding boxes
                pHitMask = env['hitmasks']['player'][player['idx']]
                uHitmask = env['hitmasks']['pipe'][0]
                lHitmask = env['hitmasks']['pipe'][1]

                # if bird collided with upipe or lpipe
                uCol = self.pixel_collision(
                    playerRect, uRect, pHitMask, uHitmask)
                lCol = self.pixel_collision(
                    playerRect, lRect, pHitMask, lHitmask)
                # rotate player on pipe crash
                if uCol or lCol: 
                    player['rot'].update(-player['rot_vel'])
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
