
'''
Deep-Q Reinforcement Learning for Flappy Bird
File:    flappy_load.py
Author:  Whitney King
Date:    March 8, 2019

Ingests image assets, prepares them for consumption
by the game, then calculates collision boundaries 
for each of the different sprites

References:
    FlapPy Bird
    Author: Sourabh Verma
    github.com/sourabhv/FlapPyBird

    Using Deep-Q Network to Learn to Play Flappy Bird
    Author: Kevin Chen
    github.com/yenchenlin/DeepLearningFlappyBird
'''

import os
import sys
import cv2
import pygame
from skimage import io

def load():
    # image and hit mask dictionaries
    IMAGES, HITMASKS = {}, {}
    BACKGROUND       = 'assets/sprites/background-black.png'
    PIPE             = 'assets/sprites/pipe-green.png'

    # player sprites (3 positions of flap)
    PLAYER = ( # yellow bird
               'assets/sprites/yellowbird-upflap.png',
               'assets/sprites/yellowbird-midflap.png',
               'assets/sprites/yellowbird-downflap.png' )

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )
    
    # game over sprite
    IMAGES['gameover']   = pygame.image.load('assets/sprites/gameover.png')
    IMAGES['gameover'].convert_alpha()

    # base (ground) sprite
    IMAGES['base']       = pygame.image.load('assets/sprites/base.png')
    IMAGES['base'].convert_alpha()

    IMAGES['player']     = (
        pygame.image.load(PLAYER[0]).convert_alpha(),
        pygame.image.load(PLAYER[1]).convert_alpha(),
        pygame.image.load(PLAYER[2]).convert_alpha() )

    IMAGES['background'] = pygame.image.load(BACKGROUND).convert()

    # pipe sprites
    IMAGES['pipe']       = (
        pygame.transform.rotate(
            pygame.image.load(PIPE).convert_alpha(), 180),
        pygame.image.load(PIPE).convert_alpha()
    )

    # pipe hitmask
    HITMASKS['pipe']     = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1])
    )

    # player hitmask
    HITMASKS['player']   = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2])
    )

    return IMAGES, HITMASKS

def stripImage(image):
    image = cv2.cvtColor(io.imread(image), 
                         cv2.COLOR_RGBA2BGRA)
    return image

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask