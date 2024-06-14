"""
conceptual question "what colors is [shape]?"

actual question: "[shape]"
answer: "[color]"

labels: color idxes
question idxes: shape idxes
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
        self.label_strings = []
        self.question_strings = []
        color_set = set()
        shape_set = set()
        with open(join(expand(data_dir), 'descs.txt'), 'r') as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                assert int(row['n']) == len(self.label_strings)
                v = row['v']
                row = json.loads(v)
                color_name = row['color']
                shape_name = row['shape']
                self.label_strings.append(color_name)
                self.question_strings.append(shape_name)
                color_set.add(color_name)
                shape_set.add(shape_name)

        print('loaded colors and shapes')
        print(self.label_strings[:5])
        print(self.question_strings[:5])
        print(color_set)
        print(shape_set)
        self.colors = sorted(list(color_set))
        self.color2i = {color_name: i for i, color_name in enumerate(self.colors)}
        print('self.color2i', self.color2i)

        self.shapes = sorted(list(shape_set))
        self.shape2i = {shape_name: i for i, shape_name in enumerate(self.shapes)}
        print('self.shape2i', self.shape2i)

        self.labels = torch.zeros(self.N, dtype=torch.int64)
        self.questions = torch.zeros(self.N, dtype=torch.int64)
        for n in range(self.N):
            self.labels[n] = self.color2i[self.label_strings[n]]
            self.questions[n] = self.shape2i[self.question_strings[n]]
        print('self.labels[:5]', self.labels[:5])
        print('self.questions[:5]', self.questions[:5])
        self.num_classes = len(self.color2i)
        print('num_classes', self.num_classes)
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

        question_word_idxes = self.questions[idxes]
        answer_word_idxes = self.labels[idxes]
        question_texts = []
        for n in range(batch_size):
            question_texts.append(self.shapes[question_word_idxes[n]])
        label_texts = []
        for n in range(batch_size):
            label_texts.append(self.colors[answer_word_idxes[n]])

        batch = {}

        batch['labels'] = answer_word_idxes
        batch['label_texts'] = label_texts
        batch['question_word_idxes'] = question_word_idxes
        batch['question_texts'] = question_texts
        batch['feats'] = torch.from_numpy(self.feats_h5[idxes])
        batch['idxes'] = idxes
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            print('batch fetch time %.1f' % elapsed)
        if self.enable_cuda:
            # keys = list(batch.keys())
            for k in ['labels', 'feats', 'question_word_idxes']:
                batch[k] = batch[k].cuda(async=True)
        return batch


# def run(ds_ref, data_dir):
#     ds = Dataset(data_dir=data_dir, ds_ref=ds_ref)
#     b = ds.sample(batch_size=4)
#     print('question_texts', b['question_texts'])
#     print('labels', b['labels'])
#     print('feats', b['feats'].size())


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ds-ref', type=str, required=True)
#     parser.add_argument('--data-dir', type=str, default='~/data/clevr/{ds_ref}')
#     args = parser.parse_args()
#     args.data_dir = args.data_dir.format(**args.__dict__)
#     run(**args.__dict__)
