"""
convert a torch .dat file to an .h5 file (for faster loading)
"""
import json
import os
import time
from os import path
from os.path import join
import argparse

import numpy as np
import h5py

import torch

from ulfs import h5_utils


def run(in_filepath):
    with open(in_filepath, 'rb') as f:
        d = torch.load(f)
    print(f'loaded {in_filepath}')
    meta = d['meta']
    git_info = d['git_info']
    datas = d['data']
    for k, data in datas.items():
        print(k)
        for k, v in data.items():
            if isinstance(v, int):
                print(k, v)
            else:
                print(k, v.size(), v.dtype)

    assert in_filepath.endswith('.dat')
    h5_filepath = in_filepath.replace('.dat', '.h5')
    assert h5_filepath != in_filepath
    # string_dt = h5py.special_dtype(vlen=str)
    with h5py.File(h5_filepath, 'w') as h5_file:
        h5_wrapper = h5_utils.H5Wrapper(h5_file)
        h5_wrapper.store_value('meta', json.dumps(meta))
        h5_wrapper.store_value('git_info', json.dumps(git_info))
        for split, data in datas.items():
            print(split)
            for k, v in data.items():
                if isinstance(v, int):
                    print(k, v)
                    h5_wrapper.store_value(f'{split}_{k}', v)
                else:
                    print(k, v.size(), v.dtype)
                    h5_wrapper.store_tensor(f'{split}_{k}', v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-filepath', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
