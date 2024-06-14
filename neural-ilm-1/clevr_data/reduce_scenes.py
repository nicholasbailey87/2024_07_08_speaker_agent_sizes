"""
reduce the scene files into the follownig information:
- shape, color, size, material of each object
- relationships
"""
import argparse
import os
from os import path
from os.path import join
import sys
import time
import json

from ulfs.utils import expand, die


def run(ds_ref, data_dir, count):
    if count is None:
        count = len(os.listdir(join(expand(data_dir), 'scenes')))
    print('count', count)
    set_name = 'new'
    all_scenes = []
    for n in range(count):
        scene_filepath = join(data_dir, 'scenes', f'CLEVR_{set_name}_{n:06}.json')
        with open(expand(scene_filepath), 'r') as f:
            old_scene = json.load(f)
        new_scene = {}
        new_scene['relationships'] = old_scene['relationships']
        num_objects = len(old_scene['objects'])
        objects = []
        for i in range(num_objects):
            o_old = old_scene['objects'][i]
            o_new = {}
            for k in ['shape', 'color', 'size', 'material']:
                o_new[k] = o_old[k]
            objects.append(o_new)
        new_scene['n'] = n
        new_scene['objects'] = objects
        all_scenes.append(new_scene)
    out_filepath = join(data_dir, 'scenes_reduced.json')
    with open(expand(out_filepath), 'w') as f:
        json.dump(all_scenes, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-ref', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='~/data/clevr/{ds_ref}')
    parser.add_argument('--count', type=int)
    args = parser.parse_args()
    args.data_dir = args.data_dir.format(ds_ref=args.ds_ref)
    run(**args.__dict__)
