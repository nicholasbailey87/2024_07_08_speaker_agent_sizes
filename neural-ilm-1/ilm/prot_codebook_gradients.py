"""
do gradients flow into vqvae codebook?
"""
import torch
from torch import nn, optim, autograd
import numpy as np
import math, time

def run():
    num_codes = 5
    N = 7
    K = 3
    np.random.seed(123)
    torch.manual_seed(123)
    Z = torch.from_numpy(np.random.choice(num_codes, N, replace=True))
    print('Z', Z)
    codebook = nn.Parameter(torch.rand(num_codes, 3))

    # inputs = nn.Parametertorch.rand
    codebook_out = codebook[Z]
    print('codebook_out.requires_grad', codebook_out.requires_grad)
    target_out = torch.rand(N, K)
    loss = (codebook_out - target_out).pow(2).mean()
    loss.backward()
    print('codebook_out.grad', codebook_out.grad)
    print('codebook.grad', codebook.grad)

if __name__ == '__main__':
    run()
