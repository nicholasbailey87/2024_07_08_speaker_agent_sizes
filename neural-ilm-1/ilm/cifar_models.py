import torch
from torch import nn
from ulfs.nn_modules import Flatten
import torch.nn.functional as F

# from https://brandonmorris.dev/2018/06/30/wide-resnet-pytorch/
class BasicBlock(nn.Module):
    def __init__(self, inf, outf, stride, drop):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inf)
        self.conv1 = nn.Conv2d(inf, outf, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.drop = nn.Dropout(drop, inplace=True)
        self.bn2 = nn.BatchNorm2d(outf)
        self.conv2 = nn.Conv2d(outf, outf, kernel_size=3, padding=1,
                               stride=1, bias=False)
        if inf == outf:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inf), nn.ReLU(inplace=True),
                    nn.Conv2d(inf, outf, 3, padding=1, stride=stride, bias=False))

    def forward(self, x):
        x2 = self.conv1(F.relu(self.bn1(x)))
        x2 = self.drop(x2)
        x2 = self.conv2(F.relu(self.bn2(x2)))
        r = self.shortcut(x)
        return x2.add_(r)

# from https://brandonmorris.dev/2018/06/30/wide-resnet-pytorch/
class WideResNet(nn.Module):
    def __init__(self, n_grps=3, N=4, k=10, drop=0.3, first_width=16):
        super().__init__()
        layers = [nn.Conv2d(3, first_width, kernel_size=3, padding=1, bias=False)]
        # Double feature depth at each group, after the first
        widths = [first_width]
        for grp in range(n_grps):
            widths.append(first_width*(2**grp)*k)
        for grp in range(n_grps):
            layers += self._make_group(N, widths[grp], widths[grp+1],
                                       (1 if grp == 0 else 2), drop)
        layers += [
            nn.BatchNorm2d(widths[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(widths[-1], 10)
        ]
        self.features = nn.Sequential(*layers)

    def _make_group(self, N, inf, outf, stride, drop):
        group = list()
        for i in range(N):
            blk = BasicBlock(inf=(inf if i == 0 else outf), outf=outf,
                             stride=(stride if i == 0 else 1), drop=drop)
            group.append(blk)
        return group

    def forward(self, x):
        return self.features(x)
