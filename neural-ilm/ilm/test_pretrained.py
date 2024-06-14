"""
load a pretrained model, and test it

forked from pretrain_model.py
"""
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import numpy as np
import argparse
import math
import time
from os.path import expanduser as expand
from torchvision import datasets, transforms


Dataset = datasets.CIFAR10
input_units = 3072


import torch
import torch.nn as nn

from ulfs import utils

from ilm.cifar_models import WideResNet
from ilm import cifar_models
from ilm import pretrain_model


def load_model(model_path):
    with open(model_path, 'rb') as f:
        state = torch.load(f)
    enable_cuda = state['meta']['enable_cuda']
    model_class = state['meta']['teacher_model']
    if model_class in ['WideResNet']:
        Model = getattr(cifar_models, model_class)
    else:
        Model = getattr(pretrain_model, model_class)
    model = Model()
    if enable_cuda:
        model = model.cuda()

    model.load_state_dict(state['model_state'])
    print('loaded model')

    return state['meta'], model

def load_data(shuffle=True):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train = torch.utils.data.DataLoader(
        Dataset(expand('~/data'), train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=64, shuffle=shuffle, **kwargs)
    test = torch.utils.data.DataLoader(
        Dataset(expand('~/data'), train=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=64, shuffle=shuffle, **kwargs)
    return train, test

def eval_model(model, enable_cuda):
    eval_acc_sum = 0
    eval_cnt = 0
    model.eval()
    for in_batch, tgt_batch in test:
        if enable_cuda:
            in_batch = in_batch.cuda()
            tgt_batch = tgt_batch.cuda()
        with autograd.no_grad():
            logits = model(in_batch)
            _, argmax = logits.max(dim=-1)
            correct = (argmax == tgt_batch)
            acc = correct.float().mean().item()
        eval_cnt += in_batch.size(0)
        eval_acc_sum += acc * in_batch.size(0)
        print('acc', acc)
    eval_acc = eval_acc_sum / eval_cnt
    eval_err = eval_cnt - int(eval_acc_sum)

    print('eval acc %.3f' % eval_acc, 'eval err %i' % eval_err)

def run(model_path):
    train, test = load_data()
    meta, model = load_model(model_path)
    enable_cuda = meta['enable_cuda']
    eval_model(model, enable_cuda)


if __name__ == '__main__':
    utils.clean_argv()
    parser = argparse.ArgumentParser()
    # parser.add_argument('--enable-cuda', action='store_true')
    # parser.add_argument('--model-class', type=str, default='DeepModel')
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
