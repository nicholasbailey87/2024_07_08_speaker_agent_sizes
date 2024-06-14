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

from ulfs import git_info
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
        base_seed, seed_inc, images_per_batch, start_batch_index, num_batches,
        objects_per_image, echo, ref, clevr_repo, blender_dir,
        out_dir, b_side
    ):
    for d in ['images', 'scenes']:
        d = join(expand(out_dir), d)
        if not path.isdir(d):
            os.makedirs(d)
    batch_idx = start_batch_index
    meta = {
        'ref': ref,
        'base_seed': base_seed,
        'seed_inc': seed_inc,
        'images_per_batch': images_per_batch,
        'start_batch_index': start_batch_index,
        'num_batches': num_batches,
        'objects_per_image': objects_per_image
    }
    with open(join(expand(out_dir), 'meta.json'), 'w') as f:
        f.write(json.dumps(meta) + '\n')

    with open(join(expand(out_dir), 'gitinfo.json'), 'w') as f:
        f.write(json.dumps(git_info.get_gitinfo_dict()) + '\n')
    with open(join(expand(out_dir), 'gitinfo_clevr.json'), 'w') as f:
        f.write(json.dumps(git_info.get_gitinfo_dict(repo_dir=expand(clevr_repo))) + '\n')

    combos_file = 'data/CoGenT_B.json' if b_side else 'CoGenT_A.json'

    for batch_idx in range(start_batch_index, start_batch_index + num_batches):
        print('batch_idx', batch_idx)
        run_cmd(
            [
                expand(blender_dir),
                '--background',
                '--python', join(expand(clevr_repo), 'image_generation/render_images.py'),
                '--',
                '--shape_color_combos_json', join(expand(clevr_repo), 'image_generation/data', combos_file),
                '--num_images', f'{images_per_batch}',
                '--use_gpu', '1',
                '--start_idx', str(batch_idx * images_per_batch),
                '--min_objects', f'{objects_per_image}',
                '--max_objects', f'{objects_per_image}',
                '--output_image_dir', join(expand(out_dir), 'images'),
                '--output_scene_dir', join(expand(out_dir), 'scenes'),
                '--seed', str(base_seed + batch_idx * seed_inc)
            ],
            cwd=join(expand(clevr_repo), 'image_generation'),
            echo=echo
        )
    # run_cmd(['ping', '-c', '4', '127.0.0.1'])
    # while True:
    #     batch_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--echo', action='store_true')
    parser.add_argument('--b-side', action='store_true', help='use b-side combos')
    parser.add_argument('--base-seed', type=int, default=1)
    parser.add_argument('--seed-inc', type=int, default=1)
    parser.add_argument('--objects-per-image', type=int, default=2)
    parser.add_argument('--images-per-batch', type=int, default=128)
    parser.add_argument('--start-batch-index', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=450)
    parser.add_argument('--out-dir', type=str, default='~/data/clevr/{ref}')
    parser.add_argument('--clevr-repo', type=str, default='~/git/clevr-dataset-gen')
    parser.add_argument('--blender-dir', type=str, default='~/soft/blender-2.79b-linux-glibc219-x86_64/blender')
    args = parser.parse_args()
    args.out_dir = args.out_dir.format(ref=args.ref)
    run(**args.__dict__)
