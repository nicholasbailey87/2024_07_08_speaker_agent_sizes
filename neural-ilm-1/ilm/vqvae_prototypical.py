"""
we're going to try using prototypical loss with vqvae, rather than commitment loss etc

we'll use mnist data as the input, and a single latent variable
"""
import sys, os, time, datetime, csv, json, argparse
from collections import defaultdict
from os.path import join, expanduser as expand
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import torchvision as tv
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from ulfs import metrics, utils

# import ref_task.gan_dsets
# from ref_task import gan_dsets
# from ref_task.gan_dsets import MnistSampler

class MnistSampler(object):
    def __init__(self, labels=None):
        self.ds = tv.datasets.MNIST(
            download=True,
            root=expand('~/data/mnist'),
            transform=tv.transforms.ToTensor()
        )

        with open(expand('~/data/mnist/processed/training.pt'), 'rb') as f:
            d = torch.load(f)
        self.data = d[0].unsqueeze(1).float() / 255
        self.labels = d[1]
        if labels is not None:
            selected_mask = torch.zeros(self.labels.size(0), dtype=torch.uint8)
            for label in labels:
                selected_mask[self.labels == label] = 1
            # selected_idxes = (self.labels == label).nonzero().long().view(-1)
            selected_idxes = selected_mask.nonzero().long().view(-1)
            self.data = self.data[selected_idxes]
            self.labels = self.labels[selected_idxes]
        print('N', self.data.size(0))
        self.channels = self.data.size(-3)
        self.size = self.data.size(-1)

    def sample(self, batch_size):
        N = self.data.size(0)
        idxes = torch.from_numpy(np.random.choice(N, batch_size, replace=False))
        images = self.data[idxes]
        return images

    def iter_train(self, batch_size, num_examples):
        N = num_examples
        num_batches = N // batch_size
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = b_start + batch_size
            yield self.data[b_start:b_end]

def images_to_flat_for_codebook(images):
    """
    takes [N][C][H][W], converts to [N * H * W][E]
    """
    images_size = list(images.size())
    assert len(images_size) == 4
    images = images.transpose(1, 2).transpose(2, 3).contiguous().view(-1, images_size[1])
    return images_size, images

def images_for_codebook_to_4d(images_size, images_flat):
    images = images_flat.view(images_size[0], images_size[2], images_size[3], images_size[1]).transpose(2, 3).transpose(1, 2)
    return images

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x_skip = x
        x = F.relu(x)
        x = F.relu(self.bn1(self.c1(x)))
        x = self.bn2(self.c2(x))
        x = x_skip + x
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        # 64
        self.c1 = nn.Conv2d(input_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 32
        # self.c2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 16
        self.c3 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 8
        self.c4 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 4

        self.c5 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=3, padding=0)  # 1

        self.r1 = ResBlock(embedding_dim)
        self.r2 = ResBlock(embedding_dim)

        self.bn1 = nn.BatchNorm2d(embedding_dim)
        # self.bn2 = nn.BatchNorm2d(embedding_dim)
        self.bn3 = nn.BatchNorm2d(embedding_dim)
        self.bn4 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        # x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = F.relu(self.bn4(self.c4(x)))
        x = self.r1(x)
        x = self.r2(x)
        x = self.c5(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        self.r1 = ResBlock(embedding_dim)
        self.r2 = ResBlock(embedding_dim)

        self.d0 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=3, stride=3, padding=0)

        self.d1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        # self.d2 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.d4 = nn.ConvTranspose2d(embedding_dim, input_dim, kernel_size=4, stride=2, padding=1)

        self.bn0 = nn.BatchNorm2d(embedding_dim)
        self.bn1 = nn.BatchNorm2d(embedding_dim)
        # self.bn2 = nn.BatchNorm2d(embedding_dim)
        self.bn3 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn0(self.d0(x)))
        x = self.r1(x)
        x = self.r2(x)
        x = F.relu(self.bn1(self.d1(x)))
        # x = F.relu(self.bn2(self.d2(x)))
        x = F.relu(self.bn3(self.d3(x)))
        x = torch.tanh(self.d4(x))
        return x

class VQVAE(nn.Module):
    def __init__(self, num_latents, embedding_dim):
        super().__init__()
        self.codebook = nn.Parameter((torch.rand(num_latents, embedding_dim) * 0.2 - 0.1))

    def forward(self, x):
        """
        x assumed to be a matrix, with each example as a row
        """
        distances = metrics.calc_squared_euc_dist(x, self.codebook)
        _, Z = distances.min(dim=-1)
        q = self.codebook[Z]
        q_with_st = x + (q - x).detach()

        vq_loss = (q.detach() - x).pow(2).mean()
        commitment_loss = (x.detach() - q).pow(2).mean()

        return Z, q_with_st, (vq_loss, commitment_loss)

class Model(nn.Module):
    def __init__(self, input_dim, p):
        super().__init__()
        Opt = getattr(optim, p.opt)
        self.encoder = Encoder(input_dim=input_dim, embedding_dim=p.embedding_size)
        self.decoder = Decoder(input_dim=input_dim, embedding_dim=p.embedding_size)
        self.vqvae = VQVAE(num_latents=p.num_latents, embedding_dim=p.embedding_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.dropout(encoded)
        d = list(encoded.size())
        encoded_flat = encoded.transpose(1, 2).transpose(2, 3).contiguous().view(-1, d[1])
        z_flat, encoded_q_flat, (vq_loss, commitment_loss) = self.vqvae(encoded_flat)
        z = z_flat.view(d[0], d[2], d[3])
        encoded_q = encoded_q_flat.view(d[0], d[2], d[3], -1).transpose(2, 3).transpose(1, 2).contiguous()

        reconst = self.decoder(encoded_q)
        return z, reconst, (vq_loss, commitment_loss)

def run(args):
    mnist_sampler = MnistSampler()
    batch = mnist_sampler.sample(batch_size=32)
    input_dim = batch.size(1)

    model = Model(input_dim=input_dim, p=args).to(device)
    opt = optim.Adam(lr=0.001, params=model.parameters())
    crit = nn.MSELoss()

    b = 0
    last_print = time.time()
    while True:
        images = mnist_sampler.sample(batch_size=32).to(device)

        model.train()
        Z, images_reconst, (vq_loss, commitment_loss) = model(images)
        images = images[:, :, 2:26, 2:26]
        reconst_loss = crit(images_reconst, images)
        loss = reconst_loss + args.vq * vq_loss + args.commit * commitment_loss


        opt.zero_grad()
        loss.backward()
        opt.step()

        if time.time() - last_print >= 5.0:
            model.eval()
            z_counts = defaultdict(int)
            reconst_loss_sum = 0
            num_batches = 0
            for images in mnist_sampler.iter_train(batch_size=32, num_examples=1024):
                images = images.to(device)
                with autograd.no_grad():
                    # images = mnist_sampler.sample(batch_size=32).to(device)
                    Z, images_reconst, (vq_loss, commitment_loss) = model(images)
                    images = images[:, :, 2:26, 2:26]
                    reconst_loss = crit(images_reconst, images)
                    reconst_loss_sum += reconst_loss.item()
                for z in Z.view(-1).tolist():
                    z_counts[z] += 1
                num_batches += 1
            model.train()
            reconst_loss = reconst_loss_sum / num_batches
            # print('z_counts.values()[:10]', sorted(list(z_counts.values()))[:10], len(z_counts))
            print('b', b, 'reconst_loss %.3f' % reconst_loss, 'vq_loss %.3f' % vq_loss.item(),
                'commit %.3f' % commitment_loss.item(), 'nonzero', len(z_counts))
            images_and_reconst = torch.stack([images[:20], torch.clamp(images_reconst[:20], min=0)], dim=1).transpose(0, 1)
            utils.save_image_grid(f'html/image_dump_{args.ref}.png', images_and_reconst, text=f'dump', text_size=12)
            last_print = time.time()

        b += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-size', type=int, default=50)
    parser.add_argument('--num-latents', type=int, default=100)
    parser.add_argument('--vq', type=float, default=1)
    parser.add_argument('--commit', type=float, default=0.25)
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--ref', type=str, required=True)
    args = parser.parse_args()
    run(args)
