"""
give a log file, it'll give you the code from that logfile, by doing:
- clone the repo to a folder, let's say /tmp/repo (removing that folder if already existed)
  (clone is from local current repo)
- check out the commit specified in the logfile
- apply the git diff stored in hte logfile

hopefully...
"""
import os
from os import path
from os.path import join
import shutil
import argparse
import subprocess
import time
import json

from ulfs import log_utils


def run(file, ref, checkout_folder, log_dir):
    if path.isdir(checkout_folder):
        shutil.rmtree(checkout_folder)

    log_info = log_utils.find_ref(ref=ref, file=file, log_dir=log_dir)
    print('log_info', log_info)
    log_filepath = log_info['filepath']
    print('log_filepath', log_filepath)

    with open(log_filepath, 'r') as f:
        line = f.readline().strip().replace('meta:', '').strip()
        meta = json.loads(line)
    print('meta', meta)
    git_commit = meta['gitlog'].split(' ')[0]
    print('git_commit', git_commit)
    git_diff = meta['gitdiff']
    print('git_diff', git_diff)

    print(subprocess.check_output(['git', 'clone', '.', checkout_folder]).decode('utf-8'))
    print(subprocess.check_output(['git', 'checkout', git_commit], cwd=checkout_folder).decode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--log-filepath', type=str, required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--file', type=str, required=True, help='basename of script used to generate logs, eg styleprop_consolidated')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkout-folder', type=str, default='/tmp/repo', help='this folder will be removed it already exists...')
    args = parser.parse_args()
    run(**args.__dict__)
