import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import numpy as np
import argparse
import math
from os.path import expanduser as expand
from torchvision import datasets, transforms


# Dataset = datasets.MNIST
Dataset = datasets.CIFAR10
# input_units = 784
input_units = 3072


def kl(p, q, eps=1e-6):
    kl = - (p * (((q + eps) / (p + eps)).log())).sum()
    return kl


class Model(nn.Module):
    def __init__(self, input_units):
        super().__init__()
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
        # x = F.softmax(x, dim=-1)
        return x


def run(enable_cuda, lr, num_epochs, clip_grad_norm):
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

    def train_model(model):
        print('=========== train =================')
        opt = optim.Adam(lr=lr, params=model.parameters())
        crit = nn.CrossEntropyLoss()
        epoch = 0
        while True:
            epoch_loss = 0
            epoch_acc_sum = 0
            epoch_cnt = 0
            model.train()
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
            epoch_acc =epoch_acc_sum / epoch_cnt

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
            eval_acc = eval_acc_sum / eval_cnt
            eval_err = eval_cnt - int(eval_acc_sum)

            print('e=%i' % epoch, 'l=%.3f' % epoch_loss, 'acc=%.3f' % epoch_acc, 'eval acc %.3f' % eval_acc, 'eval err %i' % eval_err)
            epoch += 1
            if epoch >= num_epochs:
                print('finished epochs')
                return eval_acc
                break

    def distill(teacher, student):
        print('=========== distill =================')
        opt = optim.Adam(lr=lr, params=student.parameters())
        # crit = nn.CrossEntropyLoss()
        # crit = nn.MSELoss()
        # crit = nn.KLDivLoss(reduction='sum')
        epoch = 0
        while True:
            epoch_loss = 0
            # epoch_acc_sum = 0
            epoch_cnt = 0
            student.train()
            for in_batch, tgt_batch in train:
                if enable_cuda:
                    in_batch = in_batch.cuda()
                    tgt_batch = tgt_batch.cuda()
                logits_teacher = F.softmax(teacher(in_batch), dim=-1)
                # logits = logits
                logits_student = F.softmax(student(in_batch), dim=-1)
                # loss = crit(logits_student, logits_teacher)
                # loss = crit(logits_student.log(), logits_teacher)
                loss = kl(p=logits_teacher.detach(), q=logits_student)
                # loss = crit(logits, tgt_batch)
                opt.zero_grad()
                loss.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(student.parameters(), clip_grad_norm)
                opt.step()
                # _, argmax = logits.max(dim=-1)
                # correct = (argmax == tgt_batch)
                # acc = correct.float().mean().item()
                epoch_loss += loss.item()
                epoch_cnt += in_batch.size(0)
                # epoch_acc_sum += acc * in_batch.size(0)
            # epoch_acc =epoch_acc_sum / epoch_cnt

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

            print('e=%i' % epoch, 'l=%.3f' % epoch_loss, 'eval acc %.3f' % eval_acc, 'eval err %i' % eval_err)
            epoch += 1
            if epoch >= num_epochs:
                print('finished epochs')
                return eval_acc
                break

    student = Model(input_units=input_units)
    if enable_cuda:
        student = student.cuda()

    train_model(student)
    distill_epoch = 0
    final_distill_eval_by_distill_epoch = []
    final_eval_by_distill_epoch = []
    while True:
        print('distill_epoch %i' % distill_epoch)
        teacher = student
        student = Model(input_units=input_units)
        if enable_cuda:
            student = student.cuda()
        distill_eval = distill(teacher=teacher, student=student)
        final_distill_eval_by_distill_epoch.append(distill_eval)
        final_eval = train_model(student)
        final_eval_by_distill_epoch.append(final_eval)
        for i, final_eval in enumerate(final_eval_by_distill_epoch):
            print('    ', i, final_distill_eval_by_distill_epoch[i], final_eval)
        distill_epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--clip-grad-norm', type=float)
    args = parser.parse_args()
    run(**args.__dict__)
