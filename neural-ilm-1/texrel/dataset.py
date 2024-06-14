"""
create training examples, and test distractors, for various relations

we need to do a few things:
- come up with a way to generate relations
- (a way to test relations potentially)
- come up with a way to convert relations into examples
- come up with a way to convert relations into distrators (maybe change one thing? like, a color, a position, something like that?)
"""
import random
import time
import os
from os import path
from os.path import join

import torch
from torch import nn, optim
import numpy as np
import argparse

from colorama import init as colorama_init, Fore

from ulfs import file_utils, nn_modules, git_info
from ulfs.tensor_utils import Hadamard
from ulfs.utils import expand

from texrel import things, relations, hypothesis
from texrel.grid import Grid


class Dataset(object):
    def __init__(self, rel_space, grid_size):
        self.rel_space = rel_space
        self.thing_space = rel_space.thing_space
        self.grid_size = grid_size

    def get_grid_planes(self):
        thing_space = self.thing_space
        return thing_space.color_space.size + \
            thing_space.shape_space.size
