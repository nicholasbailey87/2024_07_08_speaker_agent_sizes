"""
try various ways of removing trigrams etc from training set, given holdout set
"""
import torch
import numpy as np
import time
import math
import argparse

from ulfs.utils import die, expand


def generate_meanings(num_meaning_types, meanings_per_type):
    N = int(math.pow(meanings_per_type, num_meaning_types))
    print('N', N)

    indices = torch.from_numpy(np.arange(N))
    print('indices', indices)

    meanings = torch.zeros(N, num_meaning_types, dtype=torch.int64)
    for t in range(num_meaning_types - 1, -1, -1):
        meanings[:, t] = indices % meanings_per_type
        indices = indices // meanings_per_type
    print('meanings\n', meanings)
    return meanings


def set_to_train_holdout(all_meanings, num_holdout):
    N = all_meanings.size(0)
    shuffled_indices = torch.from_numpy(np.random.choice(N, N, replace=False))
    all_meanings = all_meanings[shuffled_indices]
    train_set = all_meanings[num_holdout:]
    holdout_set = all_meanings[:num_holdout]
    return train_set, holdout_set


# def generate_ngrams(v, N):
#     ngrams = set()


def generate_masks(seq_len, num_zeros):
    num_masks = math.factorial(seq_len) // math.factorial(num_zeros) // math.factorial(seq_len - num_zeros)
    masks = torch.zeros(num_masks, seq_len, dtype=torch.uint8)
    print('num_masks', num_masks)
    poses = torch.zeros(num_zeros, dtype=torch.int64)
    for i in range(num_zeros):
        poses[i] = i
    # print('poses', poses)
    n = 0
    while True:
        # print('poses', poses)
        this_mask = torch.ones(seq_len, dtype=torch.uint8)
        this_mask[poses] = 0
        yield this_mask
        # print('this_mask', this_mask)
        masks[n] = this_mask
        moved_ok = False
        for j in range(num_zeros - 1, -1, -1):
            # print('j', j, 'poses[j]', poses[j].item())
            if poses[j] < seq_len - 1:
                if j == num_zeros - 1 or poses[j] < poses[j + 1] - 1:
                    moved_ok = True
                    break
        if not moved_ok:
            break
        # print('moving j', j)
        poses[j] += 1
        for i in range(j + 1, num_zeros):
            # print('i', i)
            poses[i] = poses[j] + (i - j)
        n += 1
    # print('done')
    # print('masks', masks)
    # for i in range(num_masks):
        # 


def run():
    np.random.seed(123)
    torch.manual_seed(123)

    num_meaning_types = 5
    meanings_per_type = 10

    all_meanings = generate_meanings(num_meaning_types=num_meaning_types, meanings_per_type=meanings_per_type)

    num_holdout = 5

    train_set, holdout_set = set_to_train_holdout(all_meanings, num_holdout=num_holdout)
    print('train_set', train_set)
    print('holdout_set', holdout_set)

    N_train = train_set.size(0)
    print('N_train', N_train)
    valid = torch.ones(N_train, dtype=torch.uint8)
    for num_compare_zeros in range(0, num_meaning_types):
        for compare_mask in generate_masks(seq_len=num_meaning_types, num_zeros=num_compare_zeros):
            for n in range(num_holdout):
                holdout_v = holdout_set[n]
                equal = holdout_v == train_set
                equal[:, (1 - compare_mask).nonzero()] = 1
                equal = equal.min(dim=-1)[0]
                valid[equal.nonzero()] = 0
        print('num_compare_zeros', num_compare_zeros, 'valid.sum()', valid.sum().item())
    train_set = train_set[valid.nonzero()]
    N_train = train_set.size(0)
    print('N_train', N_train)


if __name__ == '__main__':
    run()
