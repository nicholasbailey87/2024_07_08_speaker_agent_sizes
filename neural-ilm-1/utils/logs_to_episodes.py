"""
input: name of logs directory
output: for most recent ~10 logfiles, print name, and last recorded episode
"""
import os
import argparse
from os import path
from os.path import join
import json

from mylib import file_utils


def run(logdir):
    files = file_utils.get_date_ordered_files(logdir)
    for file in files[-8:]:
        filepath = join(logdir, file)
        # print(filepath)
        with open(filepath, 'r') as f:
            last_line = f.read().split('\n')[-2]
        if last_line.startswith('meta:') or last_line.strip == '':
            continue
        print(filepath, json.loads(last_line)['episode'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()
    run(**args.__dict__)
