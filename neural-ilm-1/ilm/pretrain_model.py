"""
pretrain a model, probably on CIFAR, for use with classic distillation

forked from ilm/distill_classic.py
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


# Dataset = datasets.MNIST
Dataset = datasets.CIFAR10
# input_units = 784
input_units = 3072


import torch
import torch.nn as nn

from ulfs import utils

from ilm.cifar_models import WideResNet
from ilm import cifar_models


class ConvModel(nn.Module):
    def __init__(self, size=28):
        super().__init__()
        last_channels = 3
        # size = 28
        layers = []
        # 28
        layers.append(nn.Conv2d(3, 16, kernel_size=3))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 14
        layers.append(nn.Conv2d(16, 32, kernel_size=3))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 7
        self.conv = nn.Sequential(*layers)
        self.h1 = nn.Linear(6 * 6 * 32, 50)
        self.h2 = nn.Linear(50, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        N = x.size(0)
        x = self.conv(x)
        x = x.view(N, 6 * 6 * 32)
        x = self.h1(x)
        x = self.drop(x)
        x = self.h2(x)
        return x


def run(enable_cuda, lr, num_epochs, teacher_model, ref, model_save_path, save_every_epochs):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train = torch.utils.data.DataLoader(
        Dataset(expand('~/data'), train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=64, shuffle=True, **kwargs)
    test = torch.utils.data.DataLoader(
        Dataset(expand('~/data'), train=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=64, shuffle=True, **kwargs)

    if teacher_model in ['WideResNet']:
        Teacher = getattr(cifar_models, teacher_model)
    else:
        Teacher = globals()[teacher_model]

    def save_model(model, epoch):
        save_path = model_save_path.format(epoch=epoch)
        with open(save_path, 'wb') as f:
            state = {
                'meta': {
                    'teacher_model': teacher_model,
                    'enable_cuda': enable_cuda
                },
                'model_state': model.state_dict()
            }
            torch.save(state, f)
            print('saved model')

    def train_model(model):
        save_model(model=model, epoch=0)

        print('=========== train =================')
        opt = optim.Adam(lr=lr, params=model.parameters())
        crit = nn.CrossEntropyLoss()
        epoch = 0
        while True:
            epoch_loss = 0
            epoch_acc_sum = 0
            epoch_cnt = 0
            model.train()
            # last_print = time.time()
            for in_batch, tgt_batch in train:
                if enable_cuda:
                    in_batch = in_batch.cuda()
                    tgt_batch = tgt_batch.cuda()
                logits = model(in_batch)
                loss = crit(logits, tgt_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                _, argmax = logits.max(dim=-1)
                correct = (argmax == tgt_batch)
                acc = correct.float().mean().item()
                epoch_loss += loss.item()
                epoch_cnt += in_batch.size(0)
                epoch_acc_sum += acc * in_batch.size(0)
                # if time.time() - last_print >= 5.0:
                    # print('trn_cnt', epoch_cnt, 'trn_acc %.3f' % acc)
                    # print('.', end='', flush=True)
                    # last_print = time.time()
            # print('')
            epoch_acc =epoch_acc_sum / epoch_cnt

            eval_acc_sum = 0
            eval_cnt = 0
            model.eval()
            # last_print = time.time()
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
                # if time.time() - last_print >= 5.0:
                    # print('.', end='', flush=True)
                    # print('eval_cnt', eval_cnt, 'eval_acc %.3f' % acc)
                    # last_print = time.time()
            eval_acc = eval_acc_sum / eval_cnt
            eval_err = eval_cnt - int(eval_acc_sum)

            print('e=%i' % epoch, 'l=%.3f' % epoch_loss, 'acc=%.3f' % epoch_acc, 'eval acc %.3f' % eval_acc, 'eval err %i' % eval_err)
            epoch += 1
            if (epoch % save_every_epochs) == 0:
                print('saving model...')
                save_model(model=student, epoch=epoch)
            if epoch >= num_epochs:
                print('finished epochs')
                return eval_acc
                break

    student = Teacher()
    if enable_cuda:
        student = student.cuda()

    train_model(student)


if __name__ == '__main__':
    utils.clean_argv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--enable-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--save-every-epochs', type=int, default=10)
    # parser.add_argument('--clip-grad-norm', type=float)
    parser.add_argument('--teacher-model', type=str, default='DeepModel')
    parser.add_argument('--model-save-path', type=str, default='models/{ref}_e{epoch}.pth')
    args = parser.parse_args()
    args.model_save_path = args.model_save_path.format(ref=args.ref, epoch='{epoch}')
    print('model_save_path', args.model_save_path)
    run(**args.__dict__)
