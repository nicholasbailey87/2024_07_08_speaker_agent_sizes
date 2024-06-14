import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ulfs.tensor_utils import Hadamard
from ulfs.utils import die, expand


class Texturizer(object):
    def __init__(self, num_textures, num_colors, texture_size, seed, background_noise, background_mean, background_mean_std):
        self.num_textures = num_textures
        self.num_colors = num_colors
        self.texture_size = texture_size
        self.seed = seed
        self.background_mean = background_mean
        self.background_mean_std = background_mean_std
        self.background_noise = background_noise

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.colors_emb = nn.Embedding(num_colors + 1, 3)
        self.colors_emb.weight.data[0].fill_(0)
        self.colors_emb.weight.data[1:].uniform_(0.1)
        # print('self.colors_emb.weight', self.colors_emb.weight)

        self.texture_weights = torch.zeros(num_textures + 1, texture_size * texture_size, dtype=torch.int8)
        if texture_size <= 4:
            self.texture_reprs = torch.from_numpy(np.random.choice(pow(2, texture_size * texture_size) - 1, num_textures, replace=False)) + 1
            for i in range(num_textures):
                texture_idx = self.texture_reprs[i]
                binary_rep = bin(texture_idx).replace('0b', '').rjust(texture_size * texture_size, '0')
                assert len(binary_rep) == texture_size * texture_size
                binary_rep = torch.ByteTensor([int(v) for v in list(binary_rep)])
                self.texture_weights[i + 1] = binary_rep
        else:
            """
            pow(2, 6 x 6) = lots :P  Try a different way...
            (we'll just assume that the probability of drawing the same texture is vanishingly small,
            and ignore this possibility)
            """
            for i in range(num_textures):
                self.texture_weights[i + 1] = torch.from_numpy(np.random.choice(2, texture_size * texture_size, replace=True)).byte()
        self.textures_emb = nn.Embedding(num_textures + 1, texture_size * texture_size)
        self.textures_emb.weight.data[:] = self.texture_weights

    def forward(self, texture_idxes, color_idxes, savefig=False):
        """
        textures and colors are both assumed to be LongTensor's
        containing indices. where they are 0, that's a 'None' equivalent
        the indices start from 1

        their dimensions are identical, and are assumed to be:

        * [H][W]

        The result will be:

        * [3][H][W]

        (where * can be any number of dimensions)
        """
        shape_all = list(texture_idxes.size())
        texture_size = self.texture_size
        assert list(color_idxes.size()) == shape_all
        shape = shape_all[:-2]
        H = shape_all[-2]
        W = shape_all[-1]
        # print(shape, H, W)

        texture_idxes = texture_idxes.view(-1, H, W)
        color_idxes = color_idxes.view(-1, H, W)

        # assume that texture_idxes is zero iff color_idxes is zero
        N = color_idxes.size(0)

        textures = self.textures_emb(texture_idxes)
        colors = self.colors_emb(color_idxes)

        textures = textures.view(N, H, W, texture_size, texture_size).contiguous()
        textures = textures.transpose(-3, -2)
        textures = textures.contiguous().view(N, H * texture_size, W * texture_size)

        pos_mask = (texture_idxes != 0)
        # print('pos_mask[0]', pos_mask[0])
        neg_mask = 1 - pos_mask
        # print('neg_mask[0]', neg_mask[0])

        if savefig:
            for n in range(N):
                plt.imshow(textures[n].unsqueeze(-1).expand(H * texture_size, W * texture_size, 3).detach().numpy())
                plt.savefig(f'/tmp/tex{n}.png')

                plt.imshow(colors[n].detach().numpy())
                plt.savefig(f'/tmp/col{n}.png')

        textures = textures.unsqueeze(-3).expand(N, 3, H * texture_size, W * texture_size)
        colors = colors.transpose(-1, -2).transpose(-2, -3).contiguous()
        colors = colors.unsqueeze(-1).expand(N, 3, H, W, texture_size).contiguous().view(
            N, 3, H, W * texture_size)

        colors = colors.unsqueeze(-2).expand(N, 3, H, texture_size, W * texture_size).contiguous().view(
            N, 3, H * texture_size, W * texture_size)

        if savefig:
            for n in range(N):
                plt.imshow(colors[n].transpose(-3, -2).transpose(-2, -1).detach().numpy())
                plt.savefig(f'/tmp/colb{n}.png')

        textures = textures.view(*shape, 3, H * texture_size, W * texture_size)
        colors = colors.view(*shape, 3, H * texture_size, W * texture_size)

        grids = Hadamard(colors, textures)

        background = torch.randn(*shape, 3, H * texture_size, W * texture_size) * self.background_noise
        means = self.background_mean + torch.randn(*shape, 3) * self.background_mean_std
        means = means.unsqueeze(-1).unsqueeze(-1).expand(*shape, 3, H * texture_size, W * texture_size)
        # print('means[0][0]', means[0][0])
        # print('means[1][0]', means[1][0])
        # die()
        background = background + means
        # print('background[0]', background[0])

        # expand pos_mask out over the texture_size expanded grid
        neg_mask = neg_mask.unsqueeze(-1).expand(N, H, W, texture_size).contiguous().view(
            N, H, W * texture_size)
        neg_mask = neg_mask.unsqueeze(-2).expand(N, H, texture_size, W * texture_size).contiguous().view(
            N, H * texture_size, W * texture_size)
        # print('neg_mask[0]', neg_mask[0])
        # expand pos mask over the color dimension
        # print('shape', shape)
        neg_mask = neg_mask.unsqueeze(1).expand(N, 3, H * texture_size, W * texture_size)
        neg_mask = neg_mask.view(*shape, 3, H * texture_size, W * texture_size)
        # print('neg_mask[0]', neg_mask[0])

        # print('background.size()', background.size())
        # print('neg_mask.size()', neg_mask.size())
        background = Hadamard(background, neg_mask.float())

        # print('background[0]', background[0])
        grids = grids + background

        return grids
