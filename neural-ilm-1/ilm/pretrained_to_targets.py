"""
load a pretrained model, and a dataset, and run the data through the model, to get targets
writes these to a csv files
"""
import csv
import os
from os import path
from os.path import join, expanduser as expand
import argparse
import json
import time
import torch
import numpy as np
from torch import nn, autograd

from ilm.cifar_models import WideResNet
from ilm import cifar_models
from ilm import pretrain_model, test_pretrained


def run(model_path, out_file):
    meta, model = test_pretrained.load_model(model_path=model_path)
    enable_cuda = meta['enable_cuda']
    train, test = test_pretrained.load_data(shuffle=False)

    teacher_logits_all = []
    for b, (in_batch, tgt_batch) in enumerate(train):
        if enable_cuda:
            in_batch = in_batch.cuda()
        with autograd.no_grad():
            logits_teacher = model(in_batch)
        teacher_logits_all.append(logits_teacher.detach().cpu())
    teacher_logits_all = torch.cat(teacher_logits_all)
    print('teacher_logits_all.size()', teacher_logits_all.size())
    print('finished running teacher')
    state = {
        'meta': meta,
        'cached': teacher_logits_all
    }
    torch.save(state, out_file)
    print('saved to', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--out-file', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
