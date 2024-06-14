import argparse
import os
from os import path
from os.path import join
import subprocess
import json
import time
import datetime


def run_cmd(cmd_list, tail_lines=0, echo=True):
    if echo:
        print(' '.join(cmd_list))
    res = subprocess.check_output(cmd_list).decode('utf-8')
    if echo:
        print(res)
    return res


def head(file, lines):
    return subprocess.check_output(['head', '-n', str(lines), file]).decode('utf-8')


def tail(file, lines):
    return subprocess.check_output(['tail', '-n', str(lines), file]).decode('utf-8')


def run(logdir='logs', purge_older_than_minutes=60, minimum_lines=2):
    files = run_cmd(['find', logdir, '-mmin', '+%s' % purge_older_than_minutes], echo=True)
    for file in files.split('\n'):
        file = file.strip()
        if file == '':
            continue
        num_lines = int(run_cmd(['wc', '-l', file], echo=False).strip().split()[0])
        # print(file, num_lines)
        if num_lines < minimum_lines:
            print('purging', file, num_lines)
            os.unlink(file)


if __name__ == '__main__':
    run()
