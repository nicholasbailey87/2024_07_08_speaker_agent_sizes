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


class Dataset(object):
    def __init__(self, parent, name, ds_ref, idxes):
        self.parent = parent
        self.name = name
        self.ds_ref = ds_ref
        self.assigned_idxes = idxes

        self.enable_cuda = False

        self.N = self.assigned_idxes.size(0)
        print(name, self.N)

        self.colors = self.parent.colors
        self.num_classes = self.parent.num_classes
        self.image_channels = self.parent.image_channels

    def cuda(self):
        self.enable_cuda = True
        return self

    def sample(self, batch_size):
        start_time = time.time()
        idxes = torch.from_numpy(np.random.choice(self.assigned_idxes, batch_size, replace=False))
        batch = self.parent.batch_from_idxes(idxes, enable_cuda=self.enable_cuda)
        elapsed = time.time() - start_time
        # if elapsed > 1.0:
        #     print('batch fetch time %.1f' % elapsed)

        return batch


class DatasetsBase(object):
    def __init__(self):
        pass

    def get_training_set(self):
        return Dataset(
            parent=self,
            ds_ref=self.ds_ref,
            name='train',
            idxes=self.train_idxes
        )

    def get_val_set(self):
        return Dataset(
            parent=self,
            ds_ref=self.ds_ref,
            name='val',
            idxes=self.val_idxes
        )

    def batch_from_idxes(self, idxes, enable_cuda):
        batch_size = idxes.size(0)
        idxes = sorted(idxes.tolist())

        labels = torch.from_numpy(np.random.choice(2, batch_size, replace=True))
        label_texts = [str(l.item()) for l in labels]

        gnd_color_idxes = self.ex_color_idxes[idxes]
        num_colors = len(self.colors)
        offsets = torch.from_numpy(np.random.choice(num_colors - 2, batch_size, replace=True)) + 1
        wrong_color_idxes = (gnd_color_idxes + offsets) % num_colors
        question_color_idxes = labels * gnd_color_idxes + (1 - labels) * wrong_color_idxes

        question_shape_idxes = self.ex_shape_idxes[idxes]
        question_word_idxes = torch.cat([question_color_idxes.view(1, -1), question_shape_idxes.view(1, -1)], dim=0)

        question_texts = []
        for n in range(batch_size):
            question_texts.append(self.shapes[question_shape_idxes[n]] + ' ' + self.colors[question_color_idxes[n]])

        answer_word_idxes = labels
        label_texts = []
        for n in range(batch_size):
            label_texts.append(str(answer_word_idxes[n].item()))

        batch = {}
        batch['labels'] = labels
        batch['label_texts'] = label_texts
        batch['question_word_idxes'] = question_word_idxes
        batch['question_texts'] = question_texts
        batch['feats'] = torch.from_numpy(self.feats_h5[idxes])
        batch['idxes'] = idxes
        if enable_cuda:
            for k in ['labels', 'feats', 'question_word_idxes']:
                batch[k] = batch[k].cuda(async=True)
        return batch
