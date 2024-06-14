"""
run a classic distillation, from deep model to shallow model

forked from ilm_distill.py, 5 apr 2019
"""
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import numpy as np
import argparse
import math
from os.path import expanduser as expand
from torchvision import datasets, transforms

from ulfs import tensor_utils


# Dataset = datasets.MNIST
Dataset = datasets.CIFAR10

input_units = 3072


import torch
import torch.nn as nn


def kl(p, q, eps=1e-6):
    kl = - (p * (((q + eps) / (p + eps)).log())).sum()
    return kl


class DeepModel2(nn.Module):
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

class DeepModel(nn.Module):
    def __init__(self, input_units):
        super().__init__()
        self.h1 = nn.Linear(input_units, 1200)
        self.h2 = nn.Linear(1200, 1200)
        self.h3 = nn.Linear(1200, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        N = x.size(0)
        x = x.view(N, -1)
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.h3(x)
        return x


class Model(nn.Module):
    def __init__(self, input_units=input_units):
        super().__init__()
        print('input_units', input_units)
        self.h1 = nn.Linear(input_units, 800)
        self.h2 = nn.Linear(800, 800)
        self.h3 = nn.Linear(800, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        N = x.size(0)
        x = x.view(N, -1)
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.h3(x)
        return x


def run(enable_cuda, lr, num_epochs, clip_grad_norm, teacher_model, batch_size, load_cached, soft_alpha, soft_alpha_decay, tau):
    def distill(teacher_logits_all, train_inputs, train_targets, student, soft_alpha):
        N = teacher_logits_all.size(0)
        print('N', N)
        perm_idx = torch.from_numpy(np.random.choice(N, N, replace=False))

        print('=========== distill =================')
        opt = optim.Adam(lr=lr, params=student.parameters())
        epoch = 0
        crit = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
        # while True:
            epoch_loss = 0
            epoch_cnt = 0
            student.train()
            # for b, (in_batch, tgt_batch) in enumerate(batches):
            num_batches = (N + batch_size - 1) // batch_size
            for b in range(num_batches):
                batch_idx = perm_idx[b * batch_size:(b + 1) * batch_size]
                in_batch = train_inputs[batch_idx]
                tgt_batch = train_targets[batch_idx]
                if enable_cuda:
                    in_batch = in_batch.cuda()
                    tgt_batch = tgt_batch.cuda()

                logits_student = student(in_batch)
                if teacher_logits_all is not None:
                    teacher_logits_batch = teacher_logits_all[batch_idx]
                    if enable_cuda:
                        teacher_logits_batch = teacher_logits_batch.cuda()
                    loss_soft = kl(p=tensor_utils.softmax(teacher_logits_batch, tau=tau).detach(), q=F.softmax(logits_student, dim=-1))
                loss_hard = crit(logits_student, tgt_batch)

                if soft_alpha == 0:
                    loss = loss_hard
                else:
                    assert teacher_logits_all is not None
                    loss = soft_alpha * loss_soft + (1 - soft_alpha) * loss_hard

                opt.zero_grad()
                loss.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(student.parameters(), clip_grad_norm)
                opt.step()
                epoch_loss += loss.item()
                epoch_cnt += in_batch.size(0)

            soft_alpha = max(0, soft_alpha - soft_alpha_decay)

            eval_acc_sum = 0
            eval_cnt = 0
            student.eval()
            for in_batch, tgt_batch in test:
                if enable_cuda:
                    in_batch = in_batch.cuda()
                    tgt_batch = tgt_batch.cuda()
                with autograd.no_grad():
                    logits = student(in_batch)
                    _, argmax = logits.max(dim=-1)
                    correct = (argmax == tgt_batch)
                    acc = correct.float().mean().item()
                eval_cnt += in_batch.size(0)
                eval_acc_sum += acc * in_batch.size(0)
            eval_acc = eval_acc_sum / eval_cnt
            eval_err = eval_cnt - int(eval_acc_sum)

            print('e=%i' % epoch, 'l=%.3f' % epoch_loss, 'eval acc %.3f' % eval_acc, 'eval err %i' % eval_err, 'soft_alpha %.3f' % soft_alpha)
            # epoch += 1
            # if epoch >= num_epochs:
        print('finished epochs')
        return eval_acc
        # break

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train = torch.utils.data.DataLoader(
        Dataset(expand('~/data'), train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=64, shuffle=False, **kwargs)
    test = torch.utils.data.DataLoader(
        Dataset(expand('~/data'), train=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=64, shuffle=True, **kwargs)
    train_inputs = []
    train_targets = []
    for inputs, targets in train:
        train_inputs.append(inputs)
        train_targets.append(targets)
    train_inputs = torch.cat(train_inputs)
    train_targets = torch.cat(train_targets)
    train = None

    Teacher = globals()[teacher_model]

    # if load_model:
    #     with open(load_model, 'rb') as f:
    #         state = torch.load(f)
    #         import pretrain_model
    #         Teacher = getattr(pretrain_model, state['meta']['teacher_model'])
    #         student = Teacher()
    #         enable_cuda = state['meta']['enable_cuda']
    #         if enable_cuda:
    #             student = student.cuda()
    #         student.load_state_dict(state['model_state'])
    #         print('loaded model')

    teacher_logits_all = None
    student = None
    if load_cached:
        state = torch.load(load_cached)
        teacher_logits_all = state['cached']
        print('loaded teacher_logits_all', teacher_logits_all.size())
    else:
        asdfasdf()
    # else:
    #     student = Teacher()
    #     if enable_cuda:
    #         student = student.cuda()
    #     print('created new model, of class', teacher_model)
    #     distill(teacher=None, student=student, soft_alpha=0)
    #     print('trained teacher')

    distill_epoch = 0
    final_distill_eval_by_distill_epoch = []
    # final_eval_by_distill_epoch = []
    while True:
        print('distill_epoch %i' % distill_epoch)
        teacher = student
        student = Model(input_units=input_units)
        if enable_cuda:
            student = student.cuda()

        # batches = list(train)
        if teacher is not None and teacher_logits_all is None:
            print('running teacher...')
            teacher_logits_all = []
            N = train_inputs.size(0)
            idx = list(range(N))
            num_batches = (N + batch_size - 1) // batch_size
            for b in range(num_batches):
            # for in_batch, tgt_batch in batches:
                batch_idx = perm_idx[b * batch_size:(b + 1) * batch_size]
                in_batch = train_inputs[batch_idx]
                if enable_cuda:
                    in_batch = in_batch.cuda()
                with autograd.no_grad():
                    logits_teacher = teacher(in_batch)
                teacher_logits_all.append(logits_teacher.detach().cpu())
            print('finished running teacher')

        distill_eval = distill(teacher_logits_all=teacher_logits_all, train_inputs=train_inputs, train_targets=train_targets, student=student, soft_alpha=soft_alpha)
        final_distill_eval_by_distill_epoch.append(distill_eval)
        # final_eval = train_model(student)
        # final_eval_by_distill_epoch.append(final_eval)
        for i, eval in enumerate(final_distill_eval_by_distill_epoch):
            print('    ', i, eval)
        teacher_logits_all = None
        distill_epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=2)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--clip-grad-norm', type=float)
    parser.add_argument('--batch-size', type=int, default=64)
    # parser.add_argument('--load-model', type=str, help='overrides --teacher-model')
    parser.add_argument('--load-cached', type=str, help='overrides --teacher-model')
    parser.add_argument('--teacher-model', type=str, default='DeepModel')
    parser.add_argument('--soft-alpha', type=float, default=0.5, help='how much weight to give the soft targets')
    parser.add_argument('--soft-alpha-decay', type=float, default=0, help='how much to decrease soft-alpha each epoch')
    args = parser.parse_args()
    run(**args.__dict__)
