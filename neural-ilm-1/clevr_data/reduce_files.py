"""
take a bunch of files with a tempalted name, and condense them into a single file

eg if we have file color_1 with a single color 'red', and file color_2 with a single color 'green',
we can reduce them to a single file colors.txt with 'red' and 'green', as a csv in format:

n,v
0,red
1,green
"""

import argparse
import os
import csv
from os import path
from os.path import join
import json
import time

from ulfs.utils import expand, die


def run(in_dir, name_template, out_file, count):
    with open(expand(out_file), 'w') as f_out:
        dict_writer = csv.DictWriter(f_out, fieldnames=['n', 'v'])
        dict_writer.writeheader()
        for n in range(count):
            with open(join(expand(in_dir), name_template.format(i=n)), 'r') as f:
                entry = f.read().strip()
                dict_writer.writerow({'n': n, 'v': entry})
                    # f.out.write(entry + '\n')
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--name-template', type=str, required=True)
    parser.add_argument('--out-file', type=str, default='{in_dir}.txt')
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    args.out_file = args.out_file.format(**args.__dict__)
    run(**args.__dict__)
