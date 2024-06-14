"""
print metadata from a dataset
"""
import json
import os
from os import path
import argparse
from os.path import join

import torch


def run(filepath):
    with open(filepath, 'rb') as f:
        d = torch.load(f, map_location='cpu')
    print('d.keys()', d.keys())
    print('d[params]', d['params'])
    print(filepath, d['params'].ref, d['params'].data_ref)
    # print(d['meta'])
    # print(d['params'])
    # print('d.keys()', d.keys())
    # for k, v in d['params'].__dict__.items():
        # print(k, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
