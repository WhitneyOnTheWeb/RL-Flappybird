
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
import random
from skimage import io
from PIL import Image


def load():
    # image and hit mask dictionaries
    IMAGES, HITMASKS = {}, {}

    # set path for OS environment
    if os.name == 'Linux':
        PATH = str(os.getcwd()) + '/game/assets/sprites/'
    else:
        PATH = str(os.getcwd()) + '\\game\\assets\\sprites\\'
    ftype = '.png'

    # obstacle sprites
    BACKGROUND = PATH + 'background-black' + ftype
    PIPE = (
            PATH + 'pipe-green' + ftype,
            PATH + 'pipe-blue' + ftype,
            PATH + 'pipe-red' + ftype,
    )
    # player sprites (3 positions of flap)
    PLAYER = ((
            PATH + 'redbird-upflap' + ftype,
            PATH + 'redbird-midflap' + ftype,
            PATH + 'redbird-downflap' + ftype
        ),
        (
            PATH + 'yellowbird-upflap' + ftype,
            PATH + 'yellowbird-midflap' + ftype,
            PATH + 'yellowbird-downflap' + ftype
        ),# yellow bird
        (
            PATH + 'purplebird-upflap' + ftype,
            PATH + 'purplebird-midflap' + ftype,
            PATH + 'purplebird-downflap' + ftype
        ),# purple bird
        (
            PATH + 'bluebird-upflap' + ftype,
            PATH + 'bluebird-midflap' + ftype,
            PATH + 'bluebird-downflap' + ftype
    ))

# base (ground) sprite
    IMAGES['base'] = pygame.image.load(PATH + 'base' + ftype)
    IMAGES['base'].convert_alpha()

    i = random.randint(0, len(PLAYER) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYER[i][0]).convert_alpha(),
        pygame.image.load(PLAYER[i][1]).convert_alpha(),
        pygame.image.load(PLAYER[i][2]).convert_alpha())

    IMAGES['background'] = pygame.image.load(BACKGROUND).convert()

    i = random.randint(0, len(PIPE) - 1)
    # pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE[i]).convert_alpha(), 180),
        pygame.image.load(PIPE[i]).convert_alpha()
    )

    # pipe hitmask
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1])
    )

    # player hitmask
    HITMASKS['player'] = (
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
    '''returns a hitmask using an image's alpha'''
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


def rgba_to_rgb(name):
    '''ensures images are in correct format for PyGame'''
    img = Image.open(name)
    name, ftype = name.split('.')
    if len(img.split()) == 4:                # check if image is RGBA
        r, g, b, a = img.split()             # separate color channels
        img = Image.merge('RGB', (r, g, b))  # convert from RGBA to RGB
        ftype = 'bmp'                        # save as bitmap
    fname = name + '.' + ftype
    img.save(fname)


def conv_to_bmp(path):
    #print('Working Directory: ' + os.getcwd() + path)
    for i in os.listdir(path):            # for each image in folder
        rgba_to_rgb(path + i)


def color_channels(name):
    img = Image.open(name)
    bmp = [img.getpixel((x, y)) for x in range(img.width)
           for y in range(img.height)]
    return bmp
