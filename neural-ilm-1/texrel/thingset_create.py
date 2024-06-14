import argparse
import random
import json

import numpy as np

import torch

from ulfs.params import Params
from ulfs.utils import expand, die

from texrel import things


def run(p):
    torch.manual_seed(p.seed)
    random.seed(p.seed)
    np.random.seed(p.seed)

    color_space = things.ColorSpace(num_colors=p.num_colors)
    shape_space = things.ShapeSpace(num_shapes=p.num_shapes)
    thing_space = things.ThingSpace(
        color_space=color_space, shape_space=shape_space)
    num_things = thing_space.num_unique_things
    partition = [num_things - p.num_holdout_objects, p.num_holdout_objects]
    thing_space_training, thing_space_holdout = thing_space.partition(partition)

    save_dict = {}
    save_dict['meta'] = p.__dict__
    save_dict['train'] = {}
    save_dict['holdout'] = {}

    for d, thing_space in [
            (save_dict['train'], thing_space_training),
            (save_dict['holdout'], thing_space_holdout)]:
        available_items = thing_space.available_items
        print(d, len(available_items))
        items_l = []
        for i in range(len(available_items)):
            o = available_items[i]
            print('    ', o, o.color, o.shape)
            items_l.append({'color_id': o.color, 'shape_id': o.shape})
        d['available_items'] = items_l
    with open(p.out_filepath, 'w') as f:
        f.write(json.dumps(save_dict, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-colors', type=int, default=9)
    parser.add_argument('--num-shapes', type=int, default=9)
    parser.add_argument('--num-holdout-objects', type=int, default=5)
    parser.add_argument('--out-filepath', type=str, default=expand('~/data/reftask/thingset_{ref}.json'))
    args = parser.parse_args()
    args.out_filepath = args.out_filepath.format(ref=args.ref)
    run(p=Params(args.__dict__))
