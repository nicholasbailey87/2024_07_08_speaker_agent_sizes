"""
Run clevr/blender script to generate many image/scene pairs
- we'll then process these later to generate relations (no reason we cant reuse an image
across multiple relations I reckon? if anything this strengthens things, means cant just
memorize this image ==> this relation (?))

Note that to run this you need:
- to have installed Blender *the latest version*
    - using apt-get is not sufficient (at least: not on Ubuntu 16.04)
- I downloaded it from blender website, version 2.79b, and unzipped it from inside ~/soft
    - (the older version didnt work with cuda 9)
- assumes also that clevr is cloned from inside ~/git repo
- you also need to run beforehand something like:
```
echo '/persist/git/clevr-dataset-gen/image_generation' > \
    ~/soft/blender-2.79b-linux-glibc219-x86_64/2.79/python/lib/python3.5/site-packages/clevr.pth
```
"""
import subprocess
import argparse
import os
from os import path
from os.path import join
import copy
import time
import json

from ulfs import git_info, utils
from ulfs.utils import expand, die


def run_cmd(cmd_list, echo=True, cwd=None):
    f_out = open('/tmp/foo.txt', 'w')
    f_in = open('/tmp/foo.txt', 'r')
    env = copy.deepcopy(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    print(' '.join(cmd_list))
    p = subprocess.Popen(cmd_list, stderr=subprocess.STDOUT, stdout=f_out, env=env, bufsize=1, cwd=cwd)
    while p.poll() is None:
        c = f_in.read()
        if echo:
            print(c, end='')
        time.sleep(1)
    c = f_in.read()
    if echo:
        print(c, end='')
    assert p.returncode == 0


def run(
        base_seed, seed_inc, start_index, num_examples,
        min_objects, max_objects, echo, ref, clevr_repo, blender_dir,
        b_side, render_num_samples, examples_per_batch, blender_script,
        width, height, min_pixels_per_object, sizes, shapes
    ):
    # for d in ['images', 'scenes', 'rels']:
    #     d = join(expand(out_dir), d)
    #     if not path.isdir(d):
    #         os.makedirs(d)
    # batch_idx = start_batch_index
    meta = {
        'ref': ref,
        'base_seed': base_seed,
        'seed_inc': seed_inc,
        # 'images_per_batch': images_per_batch,
        'start_index': start_index,
        'num_examples': num_examples,
        'min_objects': min_objects,
        'max_objects': max_objects,
        'width': width,
        'height': height,
        'blender_script': blender_script,
        'render_num_samples': render_num_samples,
        'b_side': b_side,
        'file': __file__
    }
    out_dir = expand(f'~/data/clevr/{ref}')
    if not path.isdir(expand(out_dir)):
        os.makedirs(expand(out_dir))
    with open(join(expand(out_dir), 'meta.json'), 'w') as f:
        f.write(json.dumps(meta) + '\n')

    with open(join(expand(out_dir), 'gitinfo.json'), 'w') as f:
        f.write(json.dumps(git_info.get_gitinfo_dict()) + '\n')
    with open(join(expand(out_dir), 'gitinfo_clevr.json'), 'w') as f:
        f.write(json.dumps(git_info.get_gitinfo_dict(repo_dir=expand(clevr_repo))) + '\n')

    combos_file = 'CoGenT_B.json' if b_side else 'CoGenT_A.json'
    print('out_dir', out_dir)
    num_batches = num_examples // examples_per_batch
    for b in range(num_batches):
        start_idx = b * examples_per_batch
    # for idx in range(start_index, start_index + num_examples):
        print('start_idx', start_idx)
        cmd =  [
            expand(blender_dir),
            '--background',
            '--python', join(expand(clevr_repo), f'image_generation/{blender_script}'),
            '--',
            '--shape_color_combos_json', join(expand(clevr_repo), 'image_generation/data', combos_file),
            '--num_examples', str(examples_per_batch),
            '--use_gpu', '1',
            '--render_num_samples', str(render_num_samples),
            # '--ref', ref,
            # '--num_pos', str(num_pos),
            # '--num_neg', str(num_neg),
            '--start_idx', str(start_idx),
            '--min_objects', f'{min_objects}',
            '--max_objects', f'{max_objects}',
            '--width', f'{width}',
            '--height', f'{height}',
            '--min_pixels_per_object', f'{min_pixels_per_object}',
            '--sizes', sizes,
            '--out_dir', out_dir,
            # '--output_image_dir', join(expand(out_dir), 'images'),
            # '--output_scene_dir', join(expand(out_dir), 'scenes'),
            # '--rels_file', join(expand(out_dir), 'rels', f'rel_{idx}.txt'),
            '--seed', str(base_seed + b * seed_inc),
        ]
        if shapes is not None:
            cmd += ['--shapes', shapes]
        run_cmd(cmd, cwd=join(expand(clevr_repo), 'image_generation'), echo=echo)
    # run_cmd(['ping', '-c', '4', '127.0.0.1'])
    # while True:
    #     batch_idx += 1


if __name__ == '__main__':
    utils.clean_argv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--echo', action='store_true')
    # parser.add_argument('--num-pos', default=3, type=int)
    # parser.add_argument('--num-neg', default=3, type=int)
    parser.add_argument('--b-side', action='store_true', help='use b-side combos')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--base-seed', type=int, default=1)
    parser.add_argument('--seed-inc', type=int, default=1)
    parser.add_argument('--min_objects', type=int, default=1)
    parser.add_argument('--max_objects', type=int, default=4)
    parser.add_argument('--examples-per-batch', type=int, default=128)
    parser.add_argument('--render_num_samples', type=int, default=512)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--num_examples', type=int, default=32768)
    parser.add_argument('--blender-script', type=str, default='generate_cube_color_examples.py')
    # parser.add_argument('--out-dir', type=str, default='~/data/clevr/{ref}')
    parser.add_argument('--clevr-repo', type=str, default='~/git/clevr-dataset-gen')
    parser.add_argument('--blender-dir', type=str, default='~/soft/blender-2.79b-linux-glibc219-x86_64/blender')
    parser.add_argument('--min_pixels_per_object', type=int, default=200)
    parser.add_argument('--sizes', type=str, default='small:0.35,large:0.7')
    parser.add_argument('--shapes', type=str)
    args = parser.parse_args()
    # args.out_dir = args.out_dir.format(ref=args.ref)
    run(**args.__dict__)
