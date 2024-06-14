import os
import json
import datetime
import argparse
import subprocess


def get_date_ordered_files(target_dir):
    files = subprocess.check_output(['ls', '-rt', target_dir]).decode('utf-8').split('\n')
    files = [f for f in files if f != '']
    return files


def run(keys):
    files = get_date_ordered_files('logs')
    for file in files:
        with open(f'logs/{file}', 'r') as f:
            line = f.readline()
            num_lines = len(f.read().split('\n'))
        if line.strip() == '':
            continue
        line = line.replace('meta: ', '')
        # print('line [%s]' % line)
        meta = json.loads(line)
        # enable_food = 'food' if meta.get('enable_food', False) else ''
        # enable_cactus = 'cactus' if meta.get('enable_cactus', False) else ''
        if num_lines < 2:
            continue
        meta_str = ''
        for k in keys:
            meta_str += ' %s=%s' % (k, meta.get(k, '<UNK>'))
        # print(f'logs/{file} {num_lines} {enable_food} {enable_cactus}')
        print(f'logs/{file} lines={num_lines}' + meta_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys', type=str, default='')
    args = parser.parse_args()
    args.keys = args.keys.split(',')
    run(**args.__dict__)
