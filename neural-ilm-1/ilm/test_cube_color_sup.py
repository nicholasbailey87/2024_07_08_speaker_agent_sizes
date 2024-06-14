"""
using cube color dataset, where we have two objects, one of which is a cube, and we need to predict
the color of the cube, we're going to train this supervised, and see how that goes

results:
b=75 loss 1.474 acc 0.414 holdout_acc 0.285
b=152 loss 1.193 acc 0.570 holdout_acc 0.184
b=230 loss 0.916 acc 0.688 holdout_acc 0.324
b=308 loss 0.565 acc 0.812 holdout_acc 0.656
b=386 loss 0.706 acc 0.781 holdout_acc 0.664
b=465 loss 0.449 acc 0.852 holdout_acc 0.762
b=542 loss 0.441 acc 0.883 holdout_acc 0.742
b=621 loss 0.210 acc 0.922 holdout_acc 0.770
"""
import os, sys, csv, json, time, datetime, argparse
from os import path
from os.path import join, expanduser as expand
import torch
from torch import optim, nn, autograd
import torch.nn.functional as F
import numpy as np
import h5py

class Dataset(object):
    def __init__(self, data_dir, num_test=256):
        self.data_dir = data_dir
        with open(join(expand(data_dir), 'cube_colors_labels.csv'), 'r') as f:
            dict_reader = csv.DictReader(f)
            labels_str = [row['color'] for row in dict_reader]
        self.id_to_label = list(set(labels_str))
        self.label_to_id = {label: id for id, label in enumerate(self.id_to_label)}
        print('id_to_label', self.id_to_label)
        print('label_to_id', self.label_to_id)
        self.labels = torch.LongTensor([self.label_to_id[label] for label in labels_str])
        print('labels[:10]', self.labels[:10])
        print('labels.size()', self.labels.size())

        self.h5f = h5py.File(join(expand(data_dir), 'cube_colors_images.h5'), 'r')
        print('h5 keys', list(self.h5f.keys()))
        self.images = torch.from_numpy(self.h5f['images'][:])

        N = self.images.shape[0]
        print('N', N)
        r = np.random.RandomState(123)
        idxes = torch.from_numpy(r.choice(N, N, replace=False))
        train_idxes, test_idxes = idxes[:-num_test], idxes[-num_test:]
        self.train_images = self.images[train_idxes]
        self.test_images = self.images[test_idxes]
        print('train_images.size()', self.train_images.size())
        print('test_images.size()', self.test_images.size())
        self.train_labels, self.test_labels = self.labels[train_idxes], self.labels[test_idxes]
        print('train_labels.size()', self.train_labels.size())
        print('test_labels.size()', self.test_labels.size())
        self.N_train = self.train_images.size(0)
        self.N_test = self.test_images.size(0)
        print('N_train', self.N_train, 'N_test', self.N_test)

    def sample_batch(self, batch_size):
        idxes = torch.from_numpy(np.random.choice(self.N_train, batch_size, replace=False))
        b_images, b_labels = self.train_images[idxes], self.train_labels[idxes]
        return b_images, b_labels

    def iter_holdout(self, batch_size):
        num_batches = (self.N_test + batch_size - 1) // batch_size
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min(self.N_test, b_start + batch_size)
            b_images = self.test_images[b_start:b_end]
            b_labels = self.test_labels[b_start:b_end]
            yield b_images, b_labels

class CNN64(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)  # 32
        self.c2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)   # 32
        self.c3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)  # 16
        self.c4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # 16
        self.c5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)   # 8
        self.c6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  # 8
        self.c7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)   # 4
        self.c8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)   # 2

        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.bn4 = nn.BatchNorm2d(num_features=16)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.bn6 = nn.BatchNorm2d(num_features=32)
        self.bn7 = nn.BatchNorm2d(num_features=64)
        # self.bn8 = nn.BatchNorm2d(num_features=64)

        self.drop = nn.Dropout(dropout)

        self.fc = nn.Linear(64 * 2 * 2, 8)

    def forward(self, x):
        """
        input is 3x32x32 images
        """
        x = self.bn1(F.relu(self.c1(x)))
        x = self.bn2(F.relu(self.c2(x)))
        x = self.bn3(F.relu(self.c3(x)))
        x = self.bn4(F.relu(self.c4(x)))
        x = self.bn5(F.relu(self.c5(x)))
        x = self.bn6(F.relu(self.c6(x)))
        x = self.bn7(F.relu(self.c7(x)))
        x = self.drop(x)
        x = self.c8(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        x = self.fc(x)
        return x

def run(args):
    dataset = Dataset(data_dir=args.data_dir)
    model = CNN64()
    opt = optim.Adam(lr=0.001, params=model.parameters())
    crit = nn.CrossEntropyLoss()
    last_print = time.time()
    b = 0
    while True:
        b_images, b_labels = dataset.sample_batch(batch_size=128)
        model.train()
        logits = model(b_images)
        _, pred = logits.max(dim=-1)
        # print('pred.size()', pred.size())
        # print('pred[:5]', pred[:5])
        acc = (pred == b_labels).float().mean()
        loss = crit(logits, b_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        def run_holdout():
            acc_sum = 0
            count = 0
            model.eval()
            for b_images, b_labels in dataset.iter_holdout(batch_size=128):
                with autograd.no_grad():
                    logits = model(b_images)
                _, pred = logits.max(dim=-1)
                acc = (pred == b_labels).float().mean()
                this_count = b_images.size(0)
                acc_sum += acc * this_count
                count += this_count
            holdout_acc = acc_sum / count
            return holdout_acc

        if time.time() - last_print >= 10.0:
            holdout_acc = run_holdout()
            print('b=%i' % b, 'loss %.3f' % loss.item(), 'acc %.3f' % acc, 'holdout_acc %.3f' % holdout_acc)
            last_print = time.time()
        b += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='~/data/clevr/{ref}')
    args = parser.parse_args()
    args.data_dir = args.data_dir.format(ref=args.ref)
    run(args)
