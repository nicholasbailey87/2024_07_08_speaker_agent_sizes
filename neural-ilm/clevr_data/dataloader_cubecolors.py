"""
in this dataset, no language, just the answer indexes (which represent colors), and the input
features
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


class Dataset(object):
    def __init__(self, data_dir, ds_ref):
        self.data_dir = data_dir
        self.ds_ref = ds_ref

        self.f_h5 = h5py.File(join(expand(data_dir), 'feats.h5'))
        self.feats_h5 = self.f_h5['features']
        self.image_channels, self.image_size, image_size_ = self.feats_h5[0].shape
        assert self.image_size == image_size_

        self.N = self.feats_h5.shape[0]
        print('N', self.N, self.image_channels, self.image_size, self.image_size)

        print('loading colors...')
        self.labels_strings = []
        color_set = set()
        with open(join(expand(data_dir), 'colors.txt'), 'r') as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                assert int(row['n']) == len(self.labels_strings)
                color = row['v']
                self.labels_strings.append(color)
                color_set.add(color)

        print('loaded colors')
        print(self.labels_strings[:5])
        print(color_set)
        self.colors = sorted(list(color_set))
        self.color2i = {color_name: i for i, color_name in enumerate(self.colors)}
        print('self.color2i', self.color2i)

        self.labels = torch.zeros(self.N, dtype=torch.int64)
        for n in range(self.N):
            self.labels[n] = self.color2i[self.labels_strings[n]]
        print('self.labels[:5]', self.labels[:5])
        self.num_classes = len(self.color2i)
        self.enable_cuda = False
        self.finished_first_epoch = False
        self.first_epoch_idx = 0

    def cuda(self):
        self.enable_cuda = True
        return self

    def sample(self, batch_size):
        """
        returns dict of:
        question_idxes, question_lens, answer_idxes, images
        """
        start_time = time.time()
        if self.finished_first_epoch:
            idxes = np.random.choice(self.N, batch_size, replace=False)
            idxes = sorted(list(idxes))
        else:
            idx_start = self.first_epoch_idx
            idx_end_excl = idx_start + batch_size
            idx_end_excl = min(self.N, idx_end_excl)
            # print(idx_start, idx_end_excl)
            self.first_epoch_idx = idx_end_excl
            if self.first_epoch_idx >= self.N:
                self.finished_first_epoch = True
                print('finished first epoch')
            idxes = list(np.arange(idx_start, idx_end_excl))

        batch = {}

        batch['labels'] = self.labels[idxes]
        batch['feats'] = torch.from_numpy(self.feats_h5[idxes])
        batch['idxes'] = idxes
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            print('batch fetch time %.1f' % elapsed)
        if self.enable_cuda:
            # keys = list(batch.keys())
            for k in ['labels', 'feats']:
                batch[k] = batch[k].cuda(async=True)
        return batch


def run(ref, data_dir):
    ds = Dataset(data_dir=data_dir, ref=ref)
    b = ds.sample(batch_size=4)
    print('labels', b['labels'])
    print('feats', b['feats'].size())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='~/data/clevr/{ref}')
    args = parser.parse_args()
    args.data_dir = args.data_dir.format(**args.__dict__)
    run(**args.__dict__)
