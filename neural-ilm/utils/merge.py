"""
resyncs logs from server, then plots graph

needs config file ~/instances.yaml
"""
import argparse
import subprocess
import os
from os import path
import plot_graphs

import yaml


def get_date_ordered_files(target_dir):
    files = subprocess.check_output(['ls', '-rt', target_dir]).decode('utf-8').split('\n')
    files = [f for f in files if f != '']
    return files


def route_use_vpn(ip_address):
    cmd_list = [
        'route-use-vpn.sh',ip_address
    ]
    print(cmd_list)
    print(subprocess.check_output(cmd_list).decode('utf-8'))


def run(hostname, logfile, no_rsync, value_key, out_file, **kwargs):
    if not no_rsync:
        with open('~/instances.yaml'.replace('~', os.environ['HOME']), 'r') as f:
            config = yaml.load(f)
        ip_address = config['ip_by_name'][hostname]
        local_path = os.getcwd() + '/logs'
        remote_path = local_path.replace(os.environ['HOME'], '/home/ubuntu')

        route_use_vpn(ip_address)

        cmd_list = [
            'rsync', '-av',
            '-e', 'ssh -i %s' % config['keyfile'].replace('~/', os.environ['HOME'] + '/'),
            'ubuntu@{ip_address}:{remote_path}/'.format(remote_path=remote_path, ip_address=ip_address),
            '{local_path}/'.format(local_path=local_path)
        ]
        print(cmd_list)
        print(subprocess.check_output(cmd_list).decode('utf-8'))
    if logfile is None:
        # files = os.listdir('logs')
        files = get_date_ordered_files('logs')
        logfile = 'logs/' + files[-1]
        print(logfile)
    plot_graphs.plot_value(logfile=logfile, out_file=out_file, value_key=value_key, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, help='should be in hosts.yaml')
    # parser.add_argument('--no-download', action='store_true')
    # parser.add_argument('--name', type=str, default='log', help='used for logfile naming')
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--max-x', type=int)
    parser.add_argument('--min-y', type=float)
    parser.add_argument('--max-y', type=float)
    parser.add_argument('--no-rsync', action='store_true')
    parser.add_argument('--value-key', type=str, default='average_reward')
    parser.add_argument('--out-file', type=str, default='/tmp/out-reward.png')
    parser.add_argument('--title', type=str)
    args = parser.parse_args()
    assert args.hostname is not None or args.no_rsync
    run(**args.__dict__)
