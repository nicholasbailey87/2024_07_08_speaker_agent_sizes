"""
Given a logfile, plot a graph
"""
import json
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend('agg')


def plot_value(logfile, min_y, max_y, title, max_x, value_key, out_file):
    epoch = []
    reward = []
    with open(logfile, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                line = line.replace('meta: ', '').strip()
                meta = json.loads(line)
                continue  # skip first line
            line = line.strip()
            if line == '':
                continue
            d = json.loads(line)
            if max_x is not None and d['episode'] > max_x:
                continue
            epoch.append(int(d['episode']))
            v = float(d[value_key])
            if 'version' not in d:
                v /= meta['batch_size']
            reward.append(v)
    while len(epoch) > 200:
        new_epoch = []
        new_reward = []
        for n in range(len(epoch) // 2):
            r = (reward[n * 2] + reward[n * 2 + 1]) / 2
            e = epoch[n] * 2
            new_epoch.append(e)
            new_reward.append(r)
        epoch = new_epoch
        reward = new_reward
    if min_y is None:
        min_y = 0
    if max_y is not None:
        plt.ylim([min_y, max_y])
    plt.plot(np.array(epoch), reward, label=value_key)
    if title is not None:
        plt.title(title)
    else:
        plt.title(f'{value_key} food={meta["enable_food"]} cactus={meta["enable_cactus"]}')
    plt.xlabel('Episode')
    plt.ylabel(value_key)
    plt.legend()
    plt.savefig(out_file)


def plot_multiple_files(logfiles, min_y, max_y, title, label, max_x, value_key, out_file):
    for logfile in logfiles.split(','):
        epoch = []
        reward = []
        with open(logfile, 'r') as f:
            for n, line in enumerate(f):
                if n == 0:
                    line = line.replace('meta: ' ,'')
                    print('line', line)
                    meta = json.loads(line)
                    print('meta', meta)
                    continue
                line = line.strip()
                if line == '':
                    continue
                d = json.loads(line)
                if max_x is not None and d['episode'] > max_x:
                    continue
                epoch.append(int(d['episode']))
                reward.append(float(d[value_key]))
        while len(epoch) > 200:
            new_epoch = []
            new_reward = []
            for n in range(len(epoch) // 2):
                r = (reward[n * 2] + reward[n * 2 + 1]) / 2
                e = epoch[n] * 2
                new_epoch.append(e)
                new_reward.append(r)
            epoch = new_epoch
            reward = new_reward
        if min_y is None:
            min_y = 0
        if max_y is not None:
            plt.ylim([min_y, max_y])
        plt.plot(np.array(epoch) / 1000, reward, label=label.format(**meta))
    if title is not None:
        plt.title(title)
    plt.xlabel('Episodes of 128 games (thousands)')
    plt.ylabel(value_key.replace('_', ' '))
    plt.legend()
    plt.savefig(out_file)


def plot_multiple_keys(logfile, title, step_key, value_keys, out_file):
    # epoch = []
    rows = []
    with open(logfile, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                line = line.replace('meta: ', '').strip()
                meta = json.loads(line)
                continue  # skip first line
            line = line.strip()
            if line == '':
                continue
            d = json.loads(line)
            rows.append(d)
    average_over = 1
    while len(rows) // average_over > 200:
        average_over *= 2
    print('average_over', average_over)
    averaged_rows = []
    summed_row = {}
    this_count = 0
    epochs = []
    value_keys = value_keys.split(',')
    # this_epoch = rows[0]['episode']
    for row in rows:
        for k in value_keys:
            if k not in summed_row:
                epoch = row[step_key]
                summed_row[k] = row[k]
            else:
                summed_row[k] += row[k]
        this_count += 1
        if this_count >= average_over:
            averaged_row = {}
            for k, v in summed_row.items():
                averaged_row[k] = summed_row[k] / average_over
            averaged_rows.append(averaged_row)
            epochs.append(epoch)
            summed_row = {}
            this_count = 0

    values_by_key = defaultdict(list)
    for row in averaged_rows:
        for k in value_keys:
            # print('values_by_key[k]', values_by_key[k])
            # print('row', row)
            # print('row[k]', row[k])
            values_by_key[k].append(row[k])
    # max_by_key = {}
    # min_by_key = {}
    # for key, values in values_by_key.items():
    #     max_by_key[key] = np.max(values)
    #     min_by_key[key] = np.min(values)
    # print('max_by_key', max_by_key)
    # for key, values in values_by_key.items():
    #     # if max_by_key[key] > 0:
    #     this_max = max_by_key[key]
    #     this_min = min_by_key[key]
    #     new_values = [(v - this_min) / (this_max - this_min) for v in values]
    #     values_by_key[key] = new_values

    # for k in value_keys:
    #     plt.plot(np.array(epochs), values_by_key[k], label=k)
    # if title is not None:
    #     plt.title(title)
    for i, k in enumerate(value_keys):
        print(i, k)
        plt.subplot(len(value_keys), 1, i + 1)
        plt.plot(np.array(epochs), values_by_key[k], label=k)
        plt.xlabel(step_key)
        plt.ylabel(k)
        plt.legend()
    plt.savefig(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()

    parser_ = parsers.add_parser('plot-value')
    parser_.add_argument('--logfile', type=str, required=True)
    parser_.add_argument('--max-x', type=int)
    parser_.add_argument('--min-y', type=float)
    parser_.add_argument('--max-y', type=float)
    parser_.add_argument('--value-key', type=str, default='average_reward')
    parser_.add_argument('--title', type=str)
    parser_.set_defaults(func=plot_value)

    parser_ = parsers.add_parser('plot-multiple-files')
    parser_.add_argument('--logfiles', type=str, required=True)
    parser_.add_argument('--max-x', type=int)
    parser_.add_argument('--min-y', type=float)
    parser_.add_argument('--max-y', type=float)
    parser_.add_argument('--label', type=str, default='tau={tau}')
    parser_.add_argument('--value-key', type=str, default='average_reward')
    parser_.add_argument('--out-file', type=str, default='/tmp/out-{value_key}.png')
    parser_.add_argument('--title', type=str)
    parser_.set_defaults(func=plot_multiple_files)

    parser_ = parsers.add_parser('plot-multiple-keys')
    parser_.add_argument('--logfile', type=str, required=True)
    parser_.add_argument('--step-key', type=str, default='episode')
    parser_.add_argument('--value-keys', type=str, default='average_reward')
    parser_.add_argument('--out-file', type=str, default='tmp_plots/out.png')
    parser_.add_argument('--title', type=str)
    parser_.set_defaults(func=plot_multiple_keys)

    args = parser.parse_args()
    func = args.func
    args_dict = args.__dict__
    args.out_file = args.out_file.format(**args.__dict__)
    del args_dict['func']
    func(**args_dict)
