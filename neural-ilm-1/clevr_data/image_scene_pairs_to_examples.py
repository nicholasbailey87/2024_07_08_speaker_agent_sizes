"""
given pairs of images and scenes, generate relations, bucket them, and draw samples

we'll have a holdout set of examples that use some specific color/shape/texture combinations we didnt
see during training

hmmmm. lets use the condition used in CoGenT:

training:
- cubes are gray, blue, brown, yellow
- cylinders are red, green, purple, cyan
- spheres can have any color

testing/holdout:
- cubes are red, green, purple, cyan
- cylinders are gray, blue, brown, yellow
- spheres can have any color

(updated generation to handle the holdout set image creation)

then, in this script, we'll just take whatever data is at the specified ref (which might be holdout,
or might not be; but wont affect this script: the filtering has already been done, within that ref)

then, in this script, we can just use *everything*, no need to filter... :)
"""
import argparse
import os
from os import path
from os.path import join
import json
import time
import random
from collections import defaultdict

import numpy as np
import h5py
import torch

from ulfs.utils import expand, die


class Rels(object):
    """
    creates iterators over the possible rels,
    which we will express as sentences, since ... easy
    (just tokenize these later)
    """
    def __init__(self, object_descriptions, rels):
        self.object_descriptions = object_descriptions
        self.rels = rels
        self.num_objs = len(rels['front'])

    def __iter__(self):
        """
        we'll just enumerate all of them ...
        """
        for rel, rights_by_left in sorted(self.rels.items()):
            # print('  ', rel)
            for left, rights in enumerate(rights_by_left):
                # print('    left', left)
                for right in rights:
                    # print('      right', right)
                    yield ' '.join([self.object_descriptions[right], rel, self.object_descriptions[left]])


def object_jsons_to_descriptions(objects_l):
    text_l = []
    for object_dict in objects_l:
        o_l = []
        o_l.append(object_dict['size'])
        o_l.append(object_dict['color'])
        o_l.append(object_dict['material'])
        o_l.append(object_dict['shape'])
        text_l.append(' '.join(o_l))
    return text_l


def flip_rel(rel):
    if ' left ' in rel:
        return rel.replace(' left ', ' right ')
    if ' right ' in rel:
        return rel.replace(' right ', ' left ')
    if ' behind ' in rel:
        return rel.replace(' behind ', ' front ')
    if ' front ' in rel:
        return rel.replace(' front ', ' behind ')
    assert False, rel


def run(data_dir, ref, num_pos, num_neg):
    meta = {
        'ref': ref,
        'num_pos': num_pos,
        'num_neg': num_neg
    }
    idxes_by_caption = defaultdict(list)
    for n, file in enumerate(sorted(os.listdir(join(expand(data_dir), 'scenes')))):
        filepath = join(expand(data_dir), 'scenes', file)
        # print(n, 'filepath', filepath)
        with open(filepath, 'r') as f:
            scene = json.load(f)
            # print('scene', json.dumps(scene, indent=2))
            object_descriptions = object_jsons_to_descriptions(scene['objects'])
            # for o in object_descriptions:
            #     print(o)
            rels = scene['relationships']
            for rel in Rels(object_descriptions=object_descriptions, rels=rels):
                idxes_by_caption[rel].append(n)
                if len(idxes_by_caption[rel]) >= num_pos:
                    inv_rel = flip_rel(rel)
                    print(rel, len(idxes_by_caption[rel]), len(idxes_by_caption[inv_rel]))
                # print(rel)
        if n >= 570:
            break
    print('len(idxes_by_caption)', len(idxes_by_caption))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='~/data/clevr/{ref}')
    parser.add_argument('--num-pos', type=int, default=64)
    parser.add_argument('--num-neg', type=int, default=64)
    args = parser.parse_args()
    args.data_dir = args.data_dir.format(ref=args.ref)
    run(**args.__dict__)
