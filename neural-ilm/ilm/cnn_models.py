import torch
from torch import nn
import torch.nn.functional as F

class CNNALPooling(nn.Module):
    """
    properties:
    - output_size
    - convnet
    """
    def __init__(self, dropout, image_size, num_layers=8):
        super().__init__()
        self.layer_depths = [8, 8, 16, 16, 32, 32, 64, 64]
        # self.layer_depths = layer_depths = [32, 32, 32, 32, 32, 32, 32, 32]
        self.layer_depth = self.layer_depths[:num_layers]
        self.strides = strides = s = [2, 1, 1, 2, 1, 2, 1, 2][:num_layers]
        layer_depths = self.layer_depths

        in_channels = 3
        c_layers = []
        bn_layers = []
        convnet_layers = []
        size = image_size
        for i in range(num_layers):
            c = nn.Conv2d(in_channels=in_channels, out_channels=layer_depths[i], kernel_size=3, stride=1, padding=1)
            c_layers.append(c)
            bn = nn.BatchNorm2d(num_features=self.layer_depths[i])
            bn_layers.append(bn)
            convnet_layers.append(c)
            convnet_layers.append(nn.ReLU())
            if strides[i] == 2:
                convnet_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            convnet_layers.append(bn)
            in_channels = layer_depths[i]
            size //= strides[i]
        self.output_size = size
        # print('convnet output image size', size)
        self.convnet = nn.Sequential(*convnet_layers)
        assert size >= 1
        # print(self.convnet)

    def forward(self, x):
        x = self.convnet(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        return x

class CNNALPoolingAll(nn.Module):
    """
    properties:
    - output_size
    - convnet
    """
    def __init__(self, dropout, image_size, num_layers=8):
        super().__init__()
        self.layer_depths = [8, 8, 16, 16, 32, 32, 64, 64]
        # self.layer_depths = layer_depths = [32, 32, 32, 32, 32, 32, 32, 32]
        self.layer_depth = self.layer_depths[:num_layers]
        self.strides = strides = s = [2, 1, 1, 2, 1, 2, 1, 2][:num_layers]
        layer_depths = self.layer_depths

        in_channels = 3
        c_layers = []
        bn_layers = []
        convnet_layers = []
        size = image_size
        for i in range(num_layers):
            c = nn.Conv2d(in_channels=in_channels, out_channels=layer_depths[i], kernel_size=3, stride=1, padding=1)
            c_layers.append(c)
            bn = nn.BatchNorm2d(num_features=self.layer_depths[i])
            bn_layers.append(bn)
            convnet_layers.append(c)
            convnet_layers.append(nn.ReLU())
            if strides[i] == 2:
                convnet_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            convnet_layers.append(bn)
            in_channels = layer_depths[i]
            size //= strides[i]
        assert size >= 1
        convnet_layers.append(nn.MaxPool2d(kernel_size=size, stride=size))
        self.output_size = size
        # print('convnet output image size', size)
        self.convnet = nn.Sequential(*convnet_layers)
        # print(self.convnet)

    def forward(self, x):
        x = self.convnet(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        return x

class CNNALPoolEnd(nn.Module):
    """
    properties:
    - output_size
    - convnet
    """
    def __init__(self, dropout, image_size, num_layers=8):
        super().__init__()
        self.layer_depths = [8, 8, 16, 16, 32, 32, 64, 64]
        # self.layer_depths = layer_depths = [32, 32, 32, 32, 32, 32, 32, 32]
        self.layer_depth = self.layer_depths[:num_layers]
        self.strides = strides = s = [2, 1, 1, 2, 1, 2, 1, 2][:num_layers]
        layer_depths = self.layer_depths

        in_channels = 3
        c_layers = []
        bn_layers = []
        convnet_layers = []
        size = image_size
        for i in range(num_layers):
            c = nn.Conv2d(in_channels=in_channels, out_channels=layer_depths[i], kernel_size=3, stride=strides[i], padding=1)
            c_layers.append(c)
            bn = nn.BatchNorm2d(num_features=self.layer_depths[i])
            bn_layers.append(bn)
            convnet_layers.append(c)
            convnet_layers.append(nn.ReLU())
            convnet_layers.append(bn)
            in_channels = layer_depths[i]
            size //= strides[i]
        assert size >= 1
        convnet_layers.append(nn.MaxPool2d(kernel_size=size, stride=size))
        self.output_size = size
        self.convnet = nn.Sequential(*convnet_layers)

    def forward(self, x):
        x = self.convnet(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        return x

class CNNAL(nn.Module):
    """
    properties:
    - output_size
    - convnet
    """
    def __init__(self, dropout, image_size, num_layers=8):
        super().__init__()
        self.layer_depths = [8, 8, 16, 16, 32, 32, 64, 64]
        # self.layer_depths = layer_depths = [32, 32, 32, 32, 32, 32, 32, 32]
        self.layer_depth = self.layer_depths[:num_layers]
        self.strides = strides = s = [2, 1, 1, 2, 1, 2, 1, 2][:num_layers]
        layer_depths = self.layer_depths

        in_channels = 3
        c_layers = []
        bn_layers = []
        convnet_layers = []
        size = image_size
        for i in range(num_layers):
            c = nn.Conv2d(in_channels=in_channels, out_channels=layer_depths[i], kernel_size=3, stride=strides[i], padding=1)
            c_layers.append(c)
            bn = nn.BatchNorm2d(num_features=self.layer_depths[i])
            bn_layers.append(bn)
            convnet_layers.append(c)
            convnet_layers.append(nn.ReLU())
            convnet_layers.append(bn)
            in_channels = layer_depths[i]
            size //= strides[i]
        self.output_size = size
        self.convnet = nn.Sequential(*convnet_layers)
        assert size >= 1

    def forward(self, x):
        x = self.convnet(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        return x
