"""
conceptual question "what is the color of the rightmost shape?" (fixed task, no need for language)

actual question: ""
answer: "[color]"

labels: [color idx]
question idxes: []
"""

import os
from os import path
from os.path import join
import json
import argparse
import time
import csv
from collections import defaultdict

import torch
import numpy as np
import h5py

from ulfs import utils
from ulfs.utils import Vocab
from ulfs.utils import die, expand

from clevr_data import dataloader_base


def tokens_to_idxes(tokens_l):
    """
    create a vocab, convert the texts into indexes using this vocab
    return the vocab and the indexes

    to avoid having to store the vocab, we'll just scan all words, then sort them...
    """
    words = set()
    for tokens in tokens_l:
        for token in tokens:
            words.add(token)
    words = sorted(list(words))
    w2i = {w: i for i, w in enumerate(words)}
    idxes_l = []
    for tokens in tokens_l:
        idxes = [w2i[t] for t in tokens]
        idxes_l.append(idxes)
    return words, idxes_l
    # vocab = utils.Vocab()


def get_rightmost_object(scene):
    for i, l in enumerate(scene['relationships']['right']):
        if len(l) == 0:
            rightmost_idx = i
            break
    o = scene['objects'][rightmost_idx]
    return o


class Datasets(dataloader_base.DatasetsBase):
    def __init__(self, data_dir, ds_ref, val_size):
        self.data_dir = data_dir
        self.ds_ref = ds_ref
        self.val_size = val_size

        self.f_h5 = h5py.File(join(expand(data_dir), 'feats.h5'))
        self.feats_h5 = self.f_h5['features']
        self.image_channels, self.image_size, image_size_ = self.feats_h5[0].shape
        assert self.image_size == image_size_

        self.N = self.feats_h5.shape[0]
        print('N', self.N, self.image_channels, self.image_size, self.image_size)

        print('loading scenes...')
        # self.ex_color_names = []
        # self.ex_shape_names = []
        # color_set = set()
        # shape_set = set()
        with open(join(expand(data_dir), 'scenes_reduced.json'), 'r') as f:
            scenes = json.load(f)
            answer_tokens_l = []
            assert len(scenes) == self.N
            for n in range(self.N):
                rightmost_o = get_rightmost_object(scenes[n])
                answer_tokens_l.append([rightmost_o['color']])

        answer_vocab_list, answer_idxes_l = tokens_to_idxes(answer_tokens_l)


        print('loaded colors and shapes')
        # print(self.ex_color_names[:5])
        # print(self.ex_shape_names[:5])
        # print(color_set)
        # print(shape_set)
        self.colors = sorted(list(color_set))
        self.color2i = {color_name: i for i, color_name in enumerate(self.colors)}
        print('self.color2i', self.color2i)

        self.shapes = sorted(list(shape_set))
        self.shape2i = {shape_name: i for i, shape_name in enumerate(self.shapes)}
        print('self.shape2i', self.shape2i)

        self.ex_color_idxes = torch.zeros(self.N, dtype=torch.int64)
        self.ex_shape_idxes = torch.zeros(self.N, dtype=torch.int64)
        for n in range(self.N):
            self.ex_color_idxes[n] = self.color2i[self.ex_color_names[n]]
            self.ex_shape_idxes[n] = self.shape2i[self.ex_shape_names[n]]
        print('self.ex_color_idxes[:5]', self.ex_color_idxes[:5])
        print('self.ex_shape_idxes[:5]', self.ex_shape_idxes[:5])
        self.num_classes = 2
        print('num_classes', self.num_classes)

        self.training_size = self.N - self.val_size
        set_assignment = torch.ones(self.N, dtype=torch.int64)
        self.val_idxes = torch.from_numpy(np.random.choice(self.N, self.val_size, replace=False))
        set_assignment[self.val_idxes] = 0
        self.train_idxes = set_assignment.nonzero().view(-1).long()
        print('len val, len train, len train + val', len(self.val_idxes), len(self.train_idxes), len(self.val_idxes) + len(self.train_idxes))
