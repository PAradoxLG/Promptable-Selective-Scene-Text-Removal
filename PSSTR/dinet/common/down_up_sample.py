import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math

class DownSample(nn.Module):
    def __init__(self, channels, mode='bilinear'):
        super().__init__()
        if mode=='pixelUnshuffle':
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                nn.PixelUnshuffle(2),
            )
        elif mode=='maxpool':
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels * 2, kernel_size=1, padding=0, bias=True),
                nn.MaxPool2d(2),
            )
        elif mode=='bilinear':
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels * 2, kernel_size=1, padding=0, bias=True),
                nn.Upsample(scale_factor=0.5, mode='bilinear'),
            )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels, mode='bilinear'):
        super().__init__()
        if mode == 'pixelshuffle':
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=True),
                nn.PixelShuffle(2),
            )
        elif mode == 'bilinear':
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
            )
        elif mode == 'nearest':
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
            )

    def forward(self, x):
        return self.body(x)