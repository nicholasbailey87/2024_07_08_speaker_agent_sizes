"""
designed for use with ipynb
"""
import string
from collections import defaultdict
from os.path import join, expanduser as expand
import torch
import numpy as np

samples_dir = '../tmp'

def v_to_char(v):
    if v < 10:
        return str(v)
    v -= 10
    assert v < 26
    c = string.ascii_lowercase[v]
    return c

def print_reftask_samples(save_path, remove_common=None):
    save_path = join(samples_dir, save_path)
    with open(save_path, 'rb') as f:
        saved = torch.load(f, map_location=lambda storage, loc: storage)
    utts = saved['samples']['utts']
    labels = saved['samples']['labels']
    K = labels.size(1)
    labels_flat = labels[:, 0]
    label_max = labels.max().item()
    for k in range(1, K):
        assert label_max < 10
        labels_flat = labels_flat * 10 + labels[:, k]
    _, seq = labels_flat.sort()
    utts = utts[seq]
    labels = labels[seq]
    labels_flat = labels_flat[seq]
    N = labels.size(0)

    symbol2id = {}
    id2symbol = []
    utts_new = torch.zeros_like(utts)
    M = utts.size(1)
    for n in range(N):
        for m in range(M):
            sym = utts[n][m].item()
            if sym not in symbol2id:
                symbol2id[sym] = len(id2symbol)
                id2symbol.append(sym)
            utts_new[n][m] = symbol2id[sym]
    utts = utts_new

    last = -1
    count_by_str_by_label = {}
    strs_by_label = defaultdict(list)
    print('K', K, 'label_max', 'N', N, 'utts_max', utts.max().item())
    for n in range(N):
        label = labels_flat[n].cpu().item()
        if label != last:
            last = label
            count_by_str_by_label[label] = defaultdict(int)
        utt_str = ''.join([v_to_char(v) for v in utts[n].cpu().tolist()])
        if remove_common is not None:
            while utt_str[-1] == remove_common:
                utt_str = utt_str[:-1]
            utt_str = utt_str.ljust(M)
        count_by_str_by_label[label][utt_str] += 1
        strs_by_label[label].append(utt_str)

    if K in [1, 3]:
        for label, count_by_str in sorted(count_by_str_by_label.items()):
            # just choose one at random
            utt = np.random.choice(strs_by_label[label])
            print(label, utt)
            # for utt_str, count in sorted(count_by_str.items()):
                # print(utt_str, count)
            # print('')

    if K == 2:
        row = ''
        for x in range(label_max + 1):
            row += ' ' + str(x) + '     '
        print('  ' + row)
        for y in range(label_max + 1):
            row = ''
            for x in range(label_max + 1):
                yx = y * 10 + x
                if yx in strs_by_label:
                    # just choose one at random
                    utt = np.random.choice(strs_by_label[yx])
                    row += ' ' + utt
                else:
                    row += ' .     '
            print(str(y) + ' ' + row)
