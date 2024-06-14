#!/usr/bin/env python
import os, sys, csv, json, time, datetime, argparse
from os import path
from os.path import join, expanduser as expand
import torch
from torch import optim, nn, autograd
import torch.nn.functional as F
import numpy as np
import h5py
from ulfs import utils

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        def load_split(split, is_train):
            with open(join(expand(data_dir), f'labels_{split}.csv'), 'r') as f:
                dict_reader = csv.DictReader(f)
                shapes = set(dict_reader.fieldnames)
                shapes.remove('n')
                shapes = sorted(list(shapes))
                labels = list(dict_reader)
            if is_train:
                colors = set()
                for label in labels:
                    for shape in shapes:
                        colors.add(label[shape])
                self.id_to_color = sorted(list(colors))
                # self.id_to_color = sorted(list(set([label['cube'] for label in labels])))
                self.color_to_id = {color: id for id, color in enumerate(self.id_to_color)}
            print('self.id_to_color', self.id_to_color)

            N = len(labels)
            labels_dim = len(shapes)
            labels_t = torch.LongTensor(N, labels_dim)
            for n in range(N):
                for shape_id, shape in enumerate(shapes):
                    labels_t[n][shape_id] = self.color_to_id[labels[n][shape]]

            h5f = h5py.File(join(expand(data_dir), f'images_{split}.h5'), 'r')
            images = torch.from_numpy(h5f['images'][:])

            N = images.shape[0]

            return N, images, labels_t
        # r = np.random.RandomState(123)
        # idxes = torch.from_numpy(r.choice(N, N, replace=False))
        # train_idxes, test_idxes = idxes[:-num_test], idxes[-num_test:]
        # self.train_images = self.images[train_idxes]
        # self.test_images = self.images[test_idxes]
        # self.train_labels, self.test_labels = self.labels_t[train_idxes], self.labels_t[test_idxes]
        # self.N_train = self.train_images.size(0)
        # self.N_test = self.test_images.size(0)

        self.N_train, self.train_images, self.train_labels = load_split(split='train', is_train=True)
        self.N_test, self.test_images, self.test_labels = load_split(split='val', is_train=False)

    def train_image_count(self):
        """
        total train image count, over all N,M
        """
        return self.N_train * self.train_images.size(1)

    def sample_images(self, count):
        """
        returns a single column of images, ie each example is a single image. flat
        """
        linear_max = self.N_train * self.train_images.size(1)
        idxes_flat = torch.from_numpy(np.random.choice(linear_max, count, replace=False))
        idxes_n = idxes_flat // self.train_images.size(1)
        idxes_m = idxes_flat - idxes_n * self.train_images.size(1)
        images = self.train_images[idxes_n, idxes_m]
        # print('images.size()', images.size())
        return images

    def sample_batch(self, batch_size):
        idxes = torch.from_numpy(np.random.choice(self.N_train, batch_size, replace=False))
        b_images, b_labels = self.train_images[idxes].transpose(0, 1), self.train_labels[idxes]
        return b_images, b_labels

    def iter_holdout(self, batch_size):
        num_batches = (self.N_test + batch_size - 1) // batch_size
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min(self.N_test, b_start + batch_size)
            b_images = self.test_images[b_start:b_end].transpose(0, 1)
            b_labels = self.test_labels[b_start:b_end]
            yield b_images, b_labels

def run(args):
    dataset = Dataset(args.data_dir)
    id2label = []
    with open(join(expand(args.data_dir), 'labels_key.txt'), 'r') as f:
        for i, row in enumerate(f.read().split('\n')):
            print(i, row)
            id2label.append(row.strip())
    label2id = {label: i for i, label in enumerate(id2label)}
    print('id2label', id2label)
    print('label2id', label2id)
    for i in range(4):
        b_images, b_labels = dataset.sample_batch(batch_size=32)
        # b_images = b_images / 255
        # print('b_labels', b_labels)
        # print('b_images.size()', b_images.size())
        # b_images_flat = b_images.contiguous().view(-1)
        # sample_idxes = torch.from_numpy(np.random.choice(b_images_flat.size(0), 20))
        # print('b_images samples', b_images_flat[sample_idxes])
        # print('b_images')
        utils.save_image_grid(f'html/image_dump{i}.png', b_images.transpose(0, 1))
        for j, label in enumerate(b_labels):
            # print('label', label)
            labels_str = ','.join([dataset.id_to_color[idx] for idx in label.tolist()])
            # print(i, j, id2label[int(dataset.id_to_color[label])])
            # print(i, j, dataset.id_to_color[label])
            print(i, j, labels_str)
        print('')

    for i, (b_images, b_labels) in enumerate(dataset.iter_holdout(batch_size=32)):
        utils.save_image_grid(f'html/holdout_dump{i}.png', b_images.transpose(0, 1))
        for j, label in enumerate(b_labels):
            labels_str = ','.join([dataset.id_to_color[idx] for idx in label.tolist()])
            # print(i, j, dataset.id_to_color[label])
            print(i, j, labels_str)
        print('')

    images = dataset.sample_images(count=12)
    print('images.size()', images.size())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-family', type=str, default='clevr')
    parser.add_argument('--ds-ref', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='~/data/{data_family}/{ds_ref}')
    # parser.add_argument('--labels-key', type=str, default='~/data/{family}/{ds_ref}')
    args = parser.parse_args()
    args.data_dir = args.data_dir.format(ds_ref=args.ds_ref, data_family=args.data_family)
    run(args)
