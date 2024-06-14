from collections import defaultdict
import glob, json, math
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import tabletext

from ulfs import graphing_common
from ulfs.params import Params

# log_dir = '../logs'

def print_summary(log_dir, script, ref):
    e2e_logfiles = glob.glob(f'{log_dir}/log_{script}_{ref}_*.log')
    e2e_logfiles.sort()

    sums_by_episode = {}
    comms_dropout = None
    dropout = None
    for i, file in enumerate(e2e_logfiles):
        log_filepath = join(log_dir, file)
        num_lines = graphing_common.get_num_lines(log_filepath)
        if num_lines < 2:
            continue
        meta = graphing_common.read_meta(log_filepath)
        meta['num_lines'] = num_lines
        argv = meta['argv']
        params = Params(meta['params'])
        if params.ref != ref:
            continue
        meta['log_filepath'] = log_filepath
        if comms_dropout is None:
            comms_dropout = params.__dict__.get('comms_dropout', 0)
            dropout = params.dropout
        else:
            assert dropout == params.dropout
            assert comms_dropout == params.__dict__.get('comms_dropout', 0)
        meanings = f'{params.num_meaning_types}x{params.meanings_per_type}'
        rmlm = f'{params.model}_{params.link}_{meanings}'
        with open(log_filepath, 'r') as f_in:
            for n, line in enumerate(f_in):
                if n == 0:
                    continue
                d = json.loads(line)
                if d.get('record_type', 'ilm') != 'ilm':
                    continue
                episode = d['episode']
                sums = sums_by_episode.setdefault(episode, defaultdict(list))                
                rho = d['rho']
                holdout_acc = d['holdout_acc']
                e2e_acc = d['e2e_acc']
                sums['rho'].append(rho)
                sums['holdout_acc'].append(holdout_acc)
                sums['e2e_acc'].append(e2e_acc)
    # print('')
    episodes = []
    values_by_key = defaultdict(list)
    for episode, sums in sorted(sums_by_episode.items()):
        N = len(sums['rho'])
        res = ''
        episodes.append(episode)
        for k, v in sorted(sums.items()):
            if k == 'N':
                continue
            avg = np.mean(v).item()
            std_mean = np.std(v).item() / math.sqrt(N)
            res += f'{k}={avg:.3f}+-{std_mean:.3f} '
            values_by_key[k].append(avg)
    
    i = 1
    plt.figure(figsize=(20, 3.5))
    for k, v_l in sorted(values_by_key.items()):
        plt.subplot(1, 3, i)
        plt.plot(episodes, v_l, label=k)
        plt.legend()
        plt.title(ref)
        i += 1
    plt.show()

def print_summary_with_reruns(log_dir, refs, value_keys, scenario, step_key, res_gen, max_gen=None, script=None, scripts=None, print_meta_keys=['e2e_train_steps']):
#     print('')
#     print('*** ' + scenario + ' ***')
    print('')
    print(scenario)
    values_by_key_by_ref = {}
    episodes_by_ref = {}
    keys = set()
    printed_meta = False
    if scripts is None:
        scripts = []
    if script is not None:
        scripts.append(script)
    # print('scripts', scripts, 'ref', ref)
    for ref in refs:
        e2e_logfiles = []
        for script in scripts:
            e2e_logfiles += glob.glob(f'{log_dir}/log_{script}_{ref}_*.log')
        e2e_logfiles.sort()

        sums_by_episode = {}
        time_list_by_episode = defaultdict(list)
        comms_dropout = None
        dropout = None
        for i, file in enumerate(e2e_logfiles):
            log_filepath = join(log_dir, file)
            num_lines = graphing_common.get_num_lines(log_filepath)
            if num_lines < 2:
                continue
            meta = graphing_common.read_meta(log_filepath)
            meta['num_lines'] = num_lines
            argv = meta['argv']
            params = Params(meta['params'])
            if not printed_meta:
                p_dict = params.__dict__
                for meta_key in print_meta_keys:
                    v = None
                    sk = meta_key.split(',')
                    for k in sk:
                        if k in p_dict:
                            v = p_dict[k]
                            print(k, v)
                            break
                    if v is None:
                        print('warning: ' + meta_key + ' not found in keys ' + str(p_dict.keys()))
                printed_meta = True
            if params.ref != ref:
                continue
            meta['log_filepath'] = log_filepath
            if comms_dropout is None:
                comms_dropout = params.__dict__.get('comms_dropout', 0)
                dropout = params.dropout
            else:
                assert dropout == params.dropout
                assert comms_dropout == params.__dict__.get('comms_dropout', 0)
            with open(log_filepath, 'r') as f_in:
                for n, line in enumerate(f_in):
                    if n == 0:
                        continue
                    d = json.loads(line)
                    if d.get('record_type', 'ilm') != 'ilm':
                        continue
                    episode = d[step_key]
                    if max_gen is not None and episode > max_gen:
                        continue
                    sums = sums_by_episode.setdefault(episode, defaultdict(list))                
                    for k in value_keys:
                        v = d[k]
                        if v != v:
                            # map nan to 0
                            v = 0
                        sums[k].append(v)
                    time_list_by_episode[episode].append(d['elapsed_time'])
        episodes = []
        values_by_key = defaultdict(list)
        for episode, sums in sorted(sums_by_episode.items()):
            N = len(sums[value_keys[0]])
            res = ''
            episodes.append(episode)
            for k, v in sorted(sums.items()):
                if k == 'N':
                    continue
                avg = np.mean(v).item()
                std_mean = np.std(v).item() / math.sqrt(N)
                res += f'{k}={avg:.3f}+-{std_mean:.3f} '
                values_by_key[k].append(avg)
                keys.add(k)
        values_by_key_by_ref[ref] = values_by_key
        episodes_by_ref[ref] = episodes
    
    i = 1
    plt.figure(figsize=(20, 3.5))
    # res_by_epoch = defaultdict(dict)
    key_names = sorted(list(keys))
    tabletext_rows = [['gen', 'refs', 'count'] + key_names]
    row_values = []
    for k in key_names:
        plt.subplot(1, 3, i)
        avg_sum = 0
        avg_count = 0
        avg_values = []
        valid_refs = []
        for ref in refs:
            plt.plot(episodes_by_ref[ref], values_by_key_by_ref[ref][k], label=ref)
#             print(k, 'len(values_by_key_by_ref[ref][k])', len(values_by_key_by_ref[ref][k]))
            if len(values_by_key_by_ref[ref][k]) > res_gen:
                avg_sum += values_by_key_by_ref[ref][k][res_gen]
                avg_count += 1
                avg_values.append(values_by_key_by_ref[ref][k][res_gen])
                valid_refs.append(ref)
            # else:
            #     print('WARNING: ref ' + ref + ' missing values')
        i += 1
        if avg_count > 0:
            row_values.append('%.3f' % np.mean(avg_values) + '+/-%.3f' % (np.std(avg_values) / math.sqrt(len(avg_values))))
            # print(k + ' ' + ','.join(valid_refs) + ' count %i' % len(avg_values), '%.4' % np.mean(avg_values), '+/-%.3f' % (np.std(avg_values) / math.sqrt(len(avg_values))))
        plt.legend()
        plt.xlabel('generation')
        plt.ylabel(k)
        plt.ylim([0, 1])
        plt.title(scenario)
    plt.show()
    if len(row_values) > 0:
        table_text_row = [res_gen,  ','.join(valid_refs), len(avg_values)] + row_values
        tabletext_rows.append(table_text_row)
    # print(tabletext_rows)
    print(tabletext.to_text(tabletext_rows))
