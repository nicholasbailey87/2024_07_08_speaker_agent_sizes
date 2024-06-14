"""
draw samples from a dataset, and save them as images, so we can check the dataset isnt
completely broken :P
"""
import argparse
import json
import time
import os
from os import path
from os.path import join

import torch
import matplotlib.pyplot as plt

from ulfs.utils import expand

from texrel import dataset_runtime
from texrel.dataset_runtime import Textured3PlanesDataset


def save_image(filepath, image):
    plt.imshow(image.transpose(-3, -2).transpose(-2, -1).detach().numpy())
    plt.savefig(filepath)


def run(ds_refs, sample_holdout, ds_filepath_templ, ds_seed, ds_texture_size, ds_background_noise, ds_mean, ds_mean_std):
    ds = Textured3PlanesDataset(
        ds_refs=ds_refs,
        ds_filepath_templ=ds_filepath_templ,
        ds_seed=ds_seed,
        ds_texture_size=ds_texture_size,
        ds_background_noise=ds_background_noise,
        ds_mean=ds_mean,
        ds_mean_std=ds_mean_std
    )
    N = 32
    print('meta', json.dumps(ds.meta.__dict__, indent=2))
    b = ds.sample(batch_size=N, training=not sample_holdout)
    print('input_examples.size()', list(b['input_examples_t'].size()))
    print('receiver_examples_t.size()', list(b['receiver_examples_t'].size()))
    K = b['input_examples_t'].size(0)
    print('K', K)
    for n in range(2):
        print('example', n)
        print('label', b['labels_t'][n].item())
        print('input labels', b['input_labels_t'][:, n].tolist())
        # receiver_img = b['receiver_examples_t'][n]
        save_image(f'html/rec{n}.png', b['receiver_examples_t'][n])
        for k in range(K):
            save_image(f'html/inpN{n}_M{k}.png', b['input_examples_t'][k][n])
    print('ds_refs[:32]', b['dsrefs_t'][:32].tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-holdout', action='store_true')
    parser.add_argument('--ds-texture-size', type=int, default=4)
    parser.add_argument('--ds-seed', type=int, default=123)
    parser.add_argument('--ds-refs', type=str, required=True)
    parser.add_argument('--ds-filepath-templ', type=str, default='~/data/reftask/{ds_ref}.dat')
    parser.add_argument('--ds-background-noise', type=float, default=0, help='std of noise (with mean 0.5)')
    parser.add_argument('--ds-mean', type=float, default=0)
    parser.add_argument('--ds-mean-std', type=float, default=0)
    args = parser.parse_args()
    args.ds_refs = args.ds_refs.split(',')
    # args.data_filepath = args.data_filepath.format(ref=args.data_ref)
    run(**args.__dict__)
