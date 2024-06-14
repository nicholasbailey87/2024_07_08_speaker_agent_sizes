from ilm import test_cube_color_sup
import torch
import numpy as np

def test_dataset():
    data_dir = '~/data/clevr/dsd6_cubecolors'
    dataset = test_cube_color_sup.Dataset(data_dir)
    b_images, b_labels = dataset.sample_batch(batch_size=4)
    print('b_images[0]', b_images[0])
    print('b_labels[:10]', b_labels[:10])
    print('b_images.size()', b_images.size())
    print('b_labels.size()', b_labels.size())
