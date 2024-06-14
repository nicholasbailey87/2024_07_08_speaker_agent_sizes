"""
handles just writing a dataset to disk
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

from texrel import things, relations
from texrel import hypothesis
from texrel import thingset_runtime
from texrel.dataset import Dataset


class DatasetGenerator(object):
    def __init__(self, hypothesis_generators, ds_ref, ts_ref,
            seed, out_filepath,
            grid_size, num_distractors,
            num_pos, num_neg,
            available_preps,
            num_train, num_test
        ):
        self.ds_ref = ds_ref
        self.ts_ref = ts_ref
        self.seed = seed
        self.out_filepath = out_filepath
        self.grid_size = grid_size
        self.num_distractors = num_distractors
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.available_preps = available_preps
        self.num_train = num_train
        self.num_test = num_test

        self.available_preps = self.available_preps.split(',')
        self.num_preps = len(self.available_preps)

        self.thing_set = thingset_runtime.ThingSet.from_file(ts_ref=ts_ref)
        self.thing_space_training = self.thing_set.thing_space_by_name['train']
        self.thing_space_holdout = self.thing_set.thing_space_by_name['holdout']

        self.prep_space = relations.PrepositionSpace(
            available_preps=self.available_preps)

        self.rel_space_training = relations.RelationSpace(
            prep_space=self.prep_space,
            thing_space=self.thing_space_training)
        self.rel_space_holdout = relations.RelationSpace(
            prep_space=self.prep_space,
            thing_space=self.thing_space_holdout)

        self.hypothesis_generators_names = hypothesis_generators
        self.hypothesis_generators_training = []
        self.hypothesis_generators_holdout = []
        for hypothesis_generator_name in self.hypothesis_generators_names:
            train_hypothesis_generator_args = {}
            test_hypothesis_generator_args = {}
            if hypothesis_generator_name == 'RelationHG':
                train_hypothesis_generator_args['rel_space'] = self.rel_space_training
                train_hypothesis_generator_args['distractor_thing_space'] = self.thing_space_training
                test_hypothesis_generator_args['rel_space'] = self.rel_space_holdout
                test_hypothesis_generator_args['distractor_thing_space'] = self.thing_space_training
            else:
                train_hypothesis_generator_args['thing_space'] = self.thing_space_training
                train_hypothesis_generator_args['distractor_thing_space'] = self.thing_space_training
                test_hypothesis_generator_args['thing_space'] = self.thing_space_holdout
                test_hypothesis_generator_args['distractor_thing_space'] = self.thing_space_training
            hypothesis_generator_training = getattr(hypothesis, hypothesis_generator_name)(**train_hypothesis_generator_args)
            hypothesis_generator_holdout = getattr(hypothesis, hypothesis_generator_name)(**test_hypothesis_generator_args)
            self.hypothesis_generators_training.append(hypothesis_generator_training)
            self.hypothesis_generators_holdout.append(hypothesis_generator_holdout)
            print('hypothesis_generator', hypothesis_generator_training)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.num_shapes = self.rel_space_training.thing_space.shape_space.size
        self.num_colors = self.rel_space_training.thing_space.color_space.size
        self.num_preps = self.rel_space_training.prep_space.size
        print('num_colors', self.num_colors, 'num_shapes', self.num_shapes, 'num_preps', self.num_preps)
        self.vocab_size = 1 + self.num_colors + self.num_shapes + self.num_preps

        torch.manual_seed(self.seed + 1)

        self.dataset_training = Dataset(rel_space=self.rel_space_training, grid_size=self.grid_size)
        self.dataset_holdout = Dataset(rel_space=self.rel_space_holdout, grid_size=self.grid_size)

        self.grid_planes = self.dataset_training.get_grid_planes()

        self.batch_size = 128

    def generate_hypotheses(self, N, hgs):
        hypotheses_l = []
        for n in range(N):
            hg_idx = np.random.choice(len(hgs), 1).item()
            hg = hgs[hg_idx]
            h = hg()
            hypotheses_l.append(h)
        return hypotheses_l

    def _draw_examples(self, hgs, dataset, N):
        hypotheses_l = self.generate_hypotheses(N=N, hgs=hgs)

        input_example = hypotheses_l[0].create_positive_example(
            num_distractors=self.num_distractors, grid_size=self.grid_size)
        print('hypothesis example\n', hypotheses_l[0])
        print('input_example\n', input_example)
        seq_example = hypotheses_l[0].as_seq()
        print('seq_example', seq_example)
        max_seq_len = 0
        hypotheses_seq_l = []
        for h in hypotheses_l:
            seq, types = h.as_seq()
            max_seq_len = max(max_seq_len, seq.size(0))
            hypotheses_seq_l.append(seq)
        print('max_seq_len', max_seq_len)
        print('types', types)
        hypotheses_t = torch.zeros(max_seq_len, N, dtype=torch.int64)
        for n, seq in enumerate(hypotheses_seq_l):
            hypotheses_t[:, n] = seq
        hypotheses_t = hypotheses_t + 1
        for i, h_type in enumerate(types):
            if h_type == 'S':
                continue
            elif h_type == 'C':
                hypotheses_t[i] += self.num_shapes
            elif h_type == 'P':
                hypotheses_t[i] += self.num_shapes + self.num_colors
        print('hypotheses_t[:, :5]', hypotheses_t[:, :5])

        labels = torch.from_numpy(np.random.choice(2, N, replace=True)).byte()
        input_labels = torch.zeros(self.num_pos + self.num_neg, N, dtype=torch.uint8)
        for n in range(N):
            labels_idx = torch.from_numpy(np.random.choice(self.num_pos + self.num_neg, self.num_pos, replace=False))
            input_labels[labels_idx, n] = 1
        print('input_labels[:, :5]', input_labels[:, :5])

        input_shapes = torch.zeros(self.num_pos + self.num_neg, N, self.grid_size, self.grid_size, dtype=torch.uint8)
        input_colors = torch.zeros(self.num_pos + self.num_neg, N, self.grid_size, self.grid_size, dtype=torch.uint8)
        receiver_shapes = torch.zeros(N, self.grid_size, self.grid_size, dtype=torch.uint8)
        receiver_colors = torch.zeros(N, self.grid_size, self.grid_size, dtype=torch.uint8)

        last_print = time.time()
        for n in range(N):
            h = hypotheses_l[n]

            grid = h.create_example(label=labels[n].item(), grid_size=self.grid_size, num_distractors=self.num_distractors)
            receiver_shape, receiver_color = grid.as_shape_color_tensors(thing_space=dataset.thing_space)
            receiver_shapes[n] = receiver_shape
            receiver_colors[n] = receiver_color

            num_inputs = self.num_pos + self.num_neg
            for j in range(num_inputs):
                grid = h.create_example(label=input_labels[j][n].item(), grid_size=self.grid_size, num_distractors=self.num_distractors)
                input_shape, input_color = grid.as_shape_color_tensors(thing_space=dataset.thing_space)
                input_shapes[j, n] = input_shape
                input_colors[j, n] = input_color

            if time.time() - last_print >= 3.0:
                print('n', n)
                last_print = time.time()
        print('created shapes and colors')

        return {
            'N': N,
            'hypotheses_t': hypotheses_t,
            'input_labels': input_labels,
            'input_shapes': input_shapes,
            'input_colors': input_colors,
            'receiver_shapes': receiver_shapes,
            'receiver_colors': receiver_colors,
            'labels': labels
        }

    def generate(
            self
        ):
        """
        as of v6, this stores shapes and color tensors, and then we'll preprocess this at load time
        (probably should make a dataset class to handle loading)
        """
        train_dataset = self._draw_examples(dataset=self.dataset_training, N=self.num_train, hgs=self.hypothesis_generators_training)
        test_dataset = self._draw_examples(dataset=self.dataset_holdout, N=self.num_test, hgs=self.hypothesis_generators_holdout)
        save_dict = {
            'meta': {
                'ds_ref': self.ds_ref,
                'hypothesis_generators': self.hypothesis_generators_names,
                'num_distractors': self.num_distractors,
                'grid_size': self.grid_size,
                'num_colors': self.thing_set.meta.num_colors,
                'num_shapes': self.thing_set.meta.num_shapes,
                'available_preps': self.available_preps,
                'num_holdout_objects': self.thing_set.meta.num_holdout_objects,
                'num_pos': self.num_pos,
                'num_neg': self.num_neg,
                'vocab_size': self.vocab_size,
                'seed': self.seed
            },
            'git_info': {
                'diff': git_info.get_git_diff(),
                'log': git_info. get_git_log()
            },
            'data': {
                'train': train_dataset,
                'test': test_dataset
            }
        }
        file_utils.safe_save(self.out_filepath, save_dict)
        print('saved to ', self.out_filepath)


def run(**kwargs):
    dataset_generator = DatasetGenerator(**kwargs)
    dataset_generator.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--ds-ref', type=str, required=True)
    parser.add_argument('--ts-ref', type=str, required=True)
    parser.add_argument('--hypothesis-generators', type=str, default='RelationHG', help='comma-separated')
    parser.add_argument('--available-preps', type=str, default='LeftOf,Above', help='default is LeftOf,Above')
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--num-pos', type=int, default=3)
    parser.add_argument('--num-neg', type=int, default=3)
    parser.add_argument('--num-distractors', type=int, default=2)
    parser.add_argument('--out-filepath', type=str, default=expand('~/data/reftask/{ds_ref}.dat'))

    parser.add_argument('--num-train', type=int, default=102400)
    parser.add_argument('--num-test', type=int, default=1024)
    args = parser.parse_args()
    args.out_filepath = args.out_filepath.format(**args.__dict__)
    args.hypothesis_generators = args.hypothesis_generators.split(',')
    run(**args.__dict__)
