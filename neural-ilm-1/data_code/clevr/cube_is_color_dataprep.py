"""
Assums that you've already using clevr-dataset-gen repo to run something like:

/persist/soft/blender-2.79b-linux-glibc219-x86_64/blender --background --python generate_cube_color_examples.py -- \
    --out_dir ../output/cube_colors --min_objects 2 --max_objects 2 --num_examples 1024 --width 64 --height 64 \
    --sizes small:0.7,large:1.4 --min_pixels_per_object 40 --render_num_samples 256
(this will generate images and scene files)

What we want to do in this script is:
- reduce the images to an h5py file, and
- reduce the scenes_reduced.json to a csv file
"""
import os, sys, time, datetime, csv, json, argparse, glob
from os import path
from os.path import join, expanduser as expand
from scipy.misc import imread, imresize
import numpy as np
import h5py

def run(ref, out_dir, in_scenes_reduced, in_images, out_labels, out_images):
    print('creating', out_labels)
    scene_files = sorted(glob.glob(join(expand(out_dir), 'scenes', 'CLEVR_new_*.json')))
    image_files = sorted(glob.glob(join(expand(out_dir), 'images', '*.png')))
    N_scenes = len(scene_files)
    N_images = len(image_files)
    N = min(N_scenes, N_images)
    print('N_scenes', N_scenes, 'N_images', N_images, 'N', N)
    scene_files = scene_files[:N]
    image_files = image_files[:N]

    with open(expand(out_labels), 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=['n', 'color'])
        dict_writer.writeheader()
        for n, scene_filepath in enumerate(scene_files):
            with open(expand(scene_filepath), 'r') as f:
                scene = json.load(f)
                color = None
                for o in scene['objects']:
                    if o['shape'] == 'cube':
                        color = o['color']
                assert color is not None
                dict_writer.writerow({'n': n, 'color': color})
            n += 1
    print('done writing', out_labels)

    print('creating', out_images)
    last_print = time.time()
    with h5py.File(expand(out_images), 'w') as f:
        images_dset = None
        files = sorted(os.listdir(expand(in_images)))
        for n, img_path in enumerate(image_files):
            img = imread(expand(img_path), mode='RGB')
            img = img.transpose(2, 0, 1)[None]
            if images_dset is None:
                _, C, H, W = img.shape
                images_dset = f.create_dataset('images', (N, C, H, W), dtype=np.float32)
                print('created images dataset')
            images_dset[n] = img
            # n += 1
            if time.time() - last_print >= 5:
                print('n', n)
                last_print = time.time()
    print('created', out_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='~/data/clevr/{ref}', help='path to directory that contains images, scenes, scenes_reduced.json')
    parser.add_argument('--in-scenes-reduced', type=str, default='{out_dir}/scenes_reduced.json')
    parser.add_argument('--in-images', type=str, default='{out_dir}/images')
    parser.add_argument('--out-labels', type=str, default='{out_dir}/cube_colors_labels.csv')
    parser.add_argument('--out-images', type=str, default='{out_dir}/cube_colors_images.h5')
    args = parser.parse_args()
    args.out_dir = args.out_dir.format(**args.__dict__)
    for k, v in args.__dict__.items():
        if '{' in v:
            args.__dict__[k] = v.format(**args.__dict__)
    run(**args.__dict__)
