import os
from os import path
import json
import sys
import datetime

from ulfs import git_info


class Logger(object):
    def __init__(self, logfile, params, file):
        if not path.isdir('logs'):
            os.makedirs('logs')
        self.f = open(logfile, 'w')
        meta = {}
        meta['params'] = params.__dict__
        meta['file'] = path.splitext(path.basename(file))[0]
        meta['argv'] = sys.argv
        meta['hostname'] = os.uname().nodename
        meta['gitlog'] = git_info.get_git_log()
        meta['gitdiff'] = git_info.get_git_diff()
        meta['start_datetime'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.f.write(json.dumps(meta) + '\n')
        self.f.flush()

    def log(self, logdict, formatstr=None):
        try:
            self.f.write(json.dumps(logdict) + '\n')
        except Exception as e:
            print('exception', e)
            for k, v in logdict.items():
                print(k, type(v))
            raise e
        self.f.flush()
        if formatstr is not None:
            print_line = formatstr.format(**logdict)
            print(print_line)
