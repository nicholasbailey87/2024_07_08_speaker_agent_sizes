#!/usr/bin/env python
"""
needs:

pip install PyOpenGL

on aws do:

sudo apt-get update
sudo apt-get install -y xorg mesa-utils
nvidia-xconfig --query-gpu-info
sudo nvidia-xconfig --busid=PCI:0:30:0 --use-display-device=none --virtual=1280x1024
sudo Xorg :1

in another tab:
export DISPLAY=:1
glxinfo

... then run this script itself, ie python generate.py ...

Needs a p2 (didnt work on g3 for me)
"""

import sys, time, os, json, argparse, csv, math, random
from os import path
from os.path import join, expanduser as expand
import torch
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PIL import Image
from PIL import ImageOps

import h5py
from ulfs import utils

from data_code.objects_gl.gl_helpers import *

RED = [1, 0, 0]
GREEN = [0, 1, 0]
BLUE = [0, 0, 1]
CYAN = [0, 1, 1]
MAGENTA = [1, 0, 1]
YELLOW = [1, 1, 0]
WHITE = [1, 1, 1]
# BLACK = [0.1, 0.1, 0.1]
BLACK = [0.04, 0.04, 0.04]
color_rgbs = [RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, BLACK, WHITE]
color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']

def init(width, height):
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, height / width, 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, -1.3, -3.5)
    glRotatef(40, 1, 0, 0)

    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    glLightfv(GL_LIGHT0, GL_POSITION, [0, -3., -1, 0.])  # from top

    # glLightfv(GL_LIGHT0, GL_POSITION, [3., -3., -5, 0.])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1.0, 1, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
    glEnable(GL_LIGHT0)

def create_image(width, height, rotate, translate, object_size, shapes, colors):
    glClearColor(0, 0, 0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    ground()

    positions = []

    for i, color in enumerate(colors):
        set_color(color_rgbs[color])
        shape = shapes[i]
        # x = np.random.random_sample() * 2 - 1
        # y = np.random.random_sample() * 5
        while True:
            x = np.random.random_sample() * 2 - 1 if translate else 0
            y = np.random.random_sample() * 2 + 1.7 if translate else 2
            collide = False
            for _x, _y in positions:
                dist = math.sqrt((x - _x) * (x - _x) + (y - _y) * (y - _y))
                if dist < object_size:
                    collide = True
                    break
            if not collide:
                break
        positions.append((x, y))
        angle = np.random.randint(90) if rotate else 30
        if shape == 'cube':
            cube([x, y, 0], object_size, angle)
        elif shape == 'cylinder':
            cylinder([x, y, 0], object_size, object_size)
        elif shape == 'sphere':
            sphere([x, y, 0], object_size)

    glFlush()

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width, height), data)
    image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
    return image
    # image.save(expand(file_path), 'PNG')

def create_example(out_dir, i, num_distractors, shapes, holdout_seqs, holdout_seqs_l, is_holdout, **kwargs):
    num_shapes = len(shapes)
    if not is_holdout:
        while True:
            labels = torch.from_numpy(np.random.choice(len(color_rgbs), num_shapes, replace=True))
            labels = tuple(labels.tolist())
            # tgt_color_idxes = labels
            if labels not in holdout_seqs:
                break
    else:
        idx = np.random.randint(len(holdout_seqs_l))
        labels = holdout_seqs_l[idx]
        # labels = tgt_color_idxes

    if is_holdout:
        assert labels in holdout_seqs
        assert isinstance(labels, tuple)
    else:
        assert labels not in holdout_seqs
        assert isinstance(labels, tuple)

    images = []
    images.append(create_image(colors=labels, shapes=shapes, **kwargs))
    images.append(create_image(colors=labels, shapes=shapes, **kwargs))
    for j in range(num_distractors):
        color_idxes = list(labels)
        idx = np.random.randint(num_shapes)
        color = None
        while color is None or color == labels[idx]:
            color = np.random.randint(len(color_rgbs))
        color_idxes[idx] = color
        images.append(create_image(colors=color_idxes, shapes=shapes, **kwargs))
    return images, labels

def run(args):
    glutInit(sys.argv)

    width = args.size
    height = args.size

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"OpenGL Offscreen")
    glutHideWindow()

    init(width, height)

    if not path.isdir(expand(args.out_dir)):
        os.makedirs(expand(args.out_dir))

    with open(expand(args.meta_file), 'w') as f:
        f.write(json.dumps({
            'shapes': args.shapes,
            'ref': args.ref,
            'num_distractors': args.num_distractors,
            'rotate': args.rotate,
            'translate': not args.no_translate,
            'object_size': args.object_size,
            'size': args.size,
            'num_train': args.num_train,
            'num_holdout': args.num_holdout,
        }))

    with open(expand(args.labels_key_file), 'w') as f:
        for color in color_names:
            f.write(color + '\n')

    print('using seed', args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # need to create some holdout color sequences
    # can be any sequences in fact, as long as the color appears in at least some of the
    # non-holdout color sequences
    # we'll just pick 4 such sequences. This guarantees that at least four other sequences
    # are in the training sequences for each of these colors
    num_colors = len(color_names)
    num_shapes = len(args.shapes)
    holdout_colors = torch.from_numpy(np.random.choice(num_colors, num_shapes, replace=False))
    print('holdout_colors', holdout_colors)
    holdout_seqs = set()
    holdout_seqs_l = list()
    if num_shapes == 2:
        for p in range(num_shapes):
            holdout_color = holdout_colors[p].item()
            for j in range(num_colors):
                seq = [None] * num_shapes
                seq[p] = holdout_color
                seq[1 - p] = j
                seq = tuple(seq)
                if seq not in holdout_seqs:
                    holdout_seqs.add(seq)
                    holdout_seqs_l.append(seq)
    elif num_shapes == 3:
        for p in range(num_shapes):
            holdout_color = holdout_colors[p].item()
            q, r = 0, 1
            if p <= r:
                r += 1
            if p <= q:
                q += 1
            for i in range(num_colors):
                for j in range(num_colors):
                    seq = [None] * num_shapes
                    seq[p] = holdout_color
                    seq[q] = i
                    seq[r] = j
                    seq = tuple(seq)
                    if seq not in holdout_seqs:
                        holdout_seqs.add(seq)
                        holdout_seqs_l.append(seq)

    for seq in holdout_seqs_l:
        print(seq)
    # holdout_seqs = torch.LongTensor(sorted(list(holdout_seqs)))
    print('holdout_seqs', holdout_seqs)

    def create_dataset(split, N, is_holdout):
        print('creating split', split)
        h5file = h5py.File(expand(args.out_file.format(split=split)), 'w')
        images_h5 = h5file.create_dataset('images', (N, 2 * args.num_distractors, 3, args.size, args.size), dtype=np.float32)

        f_csv = open(expand(args.labels_file.format(split=split)), 'w')
        dict_writer = csv.DictWriter(f_csv, fieldnames=['n'] + args.shapes)
        dict_writer.writeheader()

        last_print = time.time()
        for i in range(N):
            ex, labels = create_example(
                out_dir=args.out_dir, i=i, width=width, height=height, num_distractors=args.num_distractors,
                object_size=args.object_size, rotate=args.rotate, translate=not args.no_translate,
                shapes=args.shapes, holdout_seqs=holdout_seqs, holdout_seqs_l=holdout_seqs_l,
                is_holdout=is_holdout
            )
            label_dict = {'n': i}
            for j, v in enumerate(labels):
                color_name = color_names[v]
                key_name = args.shapes[j]
                label_dict[key_name] = color_name
            dict_writer.writerow(label_dict)
            if i < 10:
                for j, image in enumerate(ex):
                    image.save(expand(join(args.out_dir, f'ex_{split}_{i}_{j}.png')), 'PNG')
                if i == 9:
                    print('saved examples')
            for j, image in enumerate(ex):
                image = np.array(image) / 255
                # if i == 0 and j == 0:
                #     print(image)
                images_h5[i, j] = np.transpose(image, [2, 0, 1])[:3]
            if time.time() - last_print >= 5.0:
                print('i', i)
                last_print = time.time()
        f_csv.close()
        print('done')

    create_dataset(split='train', N=args.num_train, is_holdout=False)
    # for a single shape, holdout set is drawn from same distribution as training set
    create_dataset(split='val', N=args.num_holdout, is_holdout=num_shapes >= 2)

    # h5file_train = h5py.File(expand(args.out_file.format(split='train')), 'w')
    # images_train_h5 = h5file_train.create_dataset('images', (args.num_examples, 2 * args.num_distractors, 3, args.size, args.size), dtype=np.float32)

    # h5file_val = h5py.File(expand(args.out_file.format(split='val')), 'w')
    # images_val_h5 = h5file_val.create_dataset('images', (args.num_examples, 2 * args.num_distractors, 3, args.size, args.size), dtype=np.float32)

    # f_csv_train = open(expand(args.labels_file.format(split='train')), 'w')
    # dict_writer_train = csv.DictWriter(f_csv_train, fieldnames=['n'] + args.shapes)
    # dict_writer_train.writeheader()

    # f_csv_val = open(expand(args.labels_file.format(split='val')), 'w')
    # dict_writer_val = csv.DictWriter(f_csv_val, fieldnames=['n'] + args.shapes)
    # dict_writer_val.writeheader()

    # last_print = time.time()
    # for i in range(args.num_train):
    #     ex, labels = create_example(
    #         out_dir=args.out_dir, i=i, width=width, height=height, num_distractors=args.num_distractors,
    #         object_size=args.object_size, rotate=args.rotate, translate=not args.no_translate,
    #         shapes=args.shapes, holdout_seqs=holdout_seqs
    #     )
    #     label_dict = {'n': i}
    #     for j, v in enumerate(labels):
    #         color_name = color_names[v]
    #         key_name = args.shapes[j]
    #         label_dict[key_name] = color_name
    #     dict_writer.writerow(label_dict)
    #     if i < 10:
    #         for j, image in enumerate(ex):
    #             image.save(expand(join(args.out_dir, f'ex_{i}_{j}.png')), 'PNG')
    #         if i == 9:
    #             print('saved examples')
    #     for j, image in enumerate(ex):
    #         image = np.array(image) / 255
    #         if i == 0 and j == 0:
    #             print(image)
    #         images_h5[i, j] = np.transpose(image, [2, 0, 1])[:3]
    #     if time.time() - last_print >= 5.0:
    #         print('i', i)
    #         last_print = time.time()
    # f_csv.close()
    # print('done')

if __name__ == '__main__':
    utils.clean_argv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--shapes', type=str, default='cube', help='comma-separated, choose from cube,sphere,cylinder')
    parser.add_argument('--num-distractors', type=int, default=4)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--no-translate', action='store_true')
    parser.add_argument('--object-size', type=float, default=0.4)
    parser.add_argument('--num-train', type=int, default=4096)
    parser.add_argument('--num-holdout', type=int, default=512)
    parser.add_argument('--out-dir', type=str, default='~/data/objects_gl/{ref}')
    parser.add_argument('--out-file', type=str, default='{out_dir}/images_{split}.h5')
    parser.add_argument('--labels-file', type=str, default='{out_dir}/labels_{split}.csv')
    parser.add_argument('--labels-key-file', type=str, default='{out_dir}/labels_key.txt')
    parser.add_argument('--meta-file', type=str, default='{out_dir}/meta.txt')
    parser.add_argument('--size', type=int, default=128)
    args = parser.parse_args()
    args.__dict__['split'] = '{split}'
    args.out_dir = args.out_dir.format(**args.__dict__)
    args.out_file = args.out_file.format(**args.__dict__)
    args.labels_file = args.labels_file.format(**args.__dict__)
    args.labels_key_file = args.labels_key_file.format(**args.__dict__)
    args.meta_file = args.meta_file.format(**args.__dict__)
    args.shapes = args.shapes.split(',')
    del args.__dict__['split']
    run(args)
