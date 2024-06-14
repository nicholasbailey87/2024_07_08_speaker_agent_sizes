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

from texrel import things, relations


class Grid(object):
    def __init__(self, size):
        self.size = size
        self.grid = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(None)
            self.grid.append(row)
        self.torch_constr = torch
        self.objects_set = set()

    # def as_tensor(self, thing_space):
    #     num_planes = thing_space.color_space.size + thing_space.shape_space.size
    #     tensor = self.torch_constr.FloatTensor(num_planes, self.size, self.size).zero_()
    #     for i in range(self.size):
    #         for j in range(self.size):
    #             if self.grid[i][j] is None:
    #                 continue
    #             o = self.grid[i][j]
    #             onehot_indices = o.as_onehot_indices(thing_space=thing_space)
    #             tensor[torch.LongTensor(onehot_indices), i, j] = 1
    #     return tensor

    def add_object(self, pos, o):
        assert self.grid[pos[0]][pos[1]] is None
        self.grid[pos[0]][pos[1]] = o
        self.objects_set.add(o)
        return self

    def as_shape_color_tensors(self, thing_space):
        shapes = self.torch_constr.LongTensor(self.size, self.size).zero_()
        colors = self.torch_constr.LongTensor(self.size, self.size).zero_()
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] is None:
                    continue
                o = self.grid[i][j]
                (s_idx, c_idx), _ = o.as_indices()
                shapes[i, j] = s_idx + 1
                colors[i, j] = c_idx + 1
        return shapes, colors

    def __repr__(self):
        res_l = []
        for i in range(self.size):
            row = ''
            for j in range(self.size):
                o = self.grid[i][j]
                if o is None:
                    row += '.'
                else:
                    fore_color = things._colors[o.color]
                    row += fore_color
                    row += things._shapes[o.shape]
                    row += Fore.RESET
            res_l.append(row)
        return '\n'.join(res_l)

    def render(self):
        print(str(self))

    def generate_available_pos(self):
        """
        returns a pos which is None at that position in the grid
        """
        pos = None
        while pos is None or self.grid[pos[0]][pos[1]] is not None:
            pos = np.random.choice(self.size, 2, replace=True)
        pos = list(pos)
        return pos
