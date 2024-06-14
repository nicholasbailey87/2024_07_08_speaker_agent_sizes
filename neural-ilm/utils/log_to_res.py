"""
converts logfile to results for pasting into a spreadsheet
"""
import os
import json
from os import path
from os.path import join
from collections import defaultdict
import argparse

from mylib import file_utils


def run(logfile, episodes, keys, average_over, logdir='logs'):
    if logfile is None:
        hostname = os.uname().nodename
        print('hostname', hostname)
        files = file_utils.get_date_ordered_files(logdir)
        files.reverse()
        for file in files:
            if hostname in file:
                filepath = join(logdir, file)
                logfile = filepath
                break
    print(logfile)
    with open(logfile, 'r') as f:
        all_rows = f.read().split('\n')
    print('num rows', len(all_rows))
    all_rows = [row.strip() for row in all_rows]
    all_rows = [json.loads(row) for row in all_rows[1:] if row != '']
    # episodes = set(episodes)
    episodes = [e for e in episodes if e < all_rows[-1]['episode']]
    all_episodes = list(episodes)
    print('episodes', episodes)
    rows = []
    buffer = []  # for average_over
    for n in range(len(all_rows) - 1, 0, -1):
        if all_rows[n]['episode'] <= episodes[-1]:
            buffer.append(all_rows[n])
            # rows.append(all_rows[n])
            if len(buffer) >= average_over:
                # summed = defaultdict(float)
                summed = {}
                for row in buffer:
                    for k in row.keys():
                        if k not in summed:
                            summed[k] = row[k]
                        else:
                            summed[k] += row[k]
                averaged = {}
                for k, v in summed.items():
                    averaged[k] = v / average_over
                rows.append(averaged)
                buffer = []
                episodes = episodes[:-1]
                if len(episodes) == 0:
                    break
    rows.reverse()
    for row in rows:
        print(row)
    print('')
    print('episode')
    for e in all_episodes:
        print(e)
    print('')

    for key in keys:
        print(key)
        for row in rows:
            print('%.3f' % row[key])
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--average-over', type=int, default=1, help='how many records to average over (for smoothing)')
    parser.add_argument('--keys', type=str, default='average_reward,steps_avg,foods_eaten_avg,cactuses_chopped_avg,exited_avg,avg_utt_len')
    parser.add_argument('--episodes', type=str, default='1000,3000,5000,8000,12000,20000,25000,30000,50000,80000,100000,130000,150000,200000')
    args = parser.parse_args()
    args.keys = args.keys.split(',')
    args.episodes = [int(v) for v in args.episodes.split(',')]
    run(**args.__dict__)
