import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from models.special_conv.dwt import DWTForward,DWTInverse
from models.common.norm import LayerNorm
from models.common.down_up_sample import DownSample

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from models.common.norm import LayerNorm


class FFN(nn.Module):
    def __init__(self, dim, expansion_factor=1, drop_out_rate=0.):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.drop1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.drop2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True, groups=hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.drop1(F.gelu(self.conv2(self.conv1(x))))
        x = self.drop2(self.conv3(x))
        return x


class SEAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x*self.net(x)

class FFTA(nn.Module):
    def __init__(self, channels, patch_size=8):
        super().__init__()

        self.patch_size = patch_size
        self.weight1 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.weight2 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.weight3 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.channel_mix = nn.Conv2d(channels*2, channels*2, kernel_size=1, bias=True)

    def forward(self, x):

        # 太慢了 不太行
        # f = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        f = torch.fft.rfft2(x.float(), norm='ortho')
        q_fft = f * self.weight1
        k_fft = f * self.weight2
        v_fft = f * self.weight3
        out = q_fft * k_fft * v_fft
        out_real = out.real
        out_imag = out.imag
        out = torch.cat([out_real, out_imag], 1)
        out = self.channel_mix(out)
        out_real, out_imag = out.chunk(2, dim=1)
        out = torch.stack((out_real, out_imag), -1)
        out = torch.view_as_complex(out)
        out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size), norm='ortho')

        # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)
        return out

class CrossFFTA(nn.Module):
    def __init__(self, channels, patch_size=8):
        super().__init__()

        self.patch_size = patch_size
        self.weight1 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.weight2 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.weight3 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.channel_mix = nn.Conv2d(channels*2, channels*2, kernel_size=1, bias=True)

    def forward(self, q, kv):

        # 太慢了 不太行
        # f = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        q = torch.fft.rfft2(q.float(), norm='ortho')
        kv = torch.fft.rfft2(kv.float(), norm='ortho')
        q_fft = q * self.weight1
        k_fft = kv * self.weight2
        v_fft = kv * self.weight3
        out = q_fft * k_fft * v_fft
        out_real = out.real
        out_imag = out.imag
        out = torch.cat([out_real, out_imag], 1)
        out = self.channel_mix(out)
        out_real, out_imag = out.chunk(2, dim=1)
        out = torch.stack((out_real, out_imag), -1)
        out = torch.view_as_complex(out)
        out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size), norm='ortho')

        # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,
                 channels,
                 patch_size=8,
                 expansion_factor=1,
                 drop_out_rate=0.2):
        super().__init__()

        self.norm1 = LayerNorm(channels)
        self.attn = FFTA(channels, patch_size)
        self.norm2 = LayerNorm(channels)
        self.ffn = FFN(channels, expansion_factor, drop_out_rate)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlockWithCrossAttention(nn.Module):
    def __init__(self, channels,patch_size=8, expansion_factor=1,drop_out_rate=0.2):
        super().__init__()

        self.norm1 = LayerNorm(channels)
        self.attn = FFTA(channels, patch_size)
        self.norm2 = LayerNorm(channels)
        self.cross_attn = CrossFFTA(channels, patch_size)
        self.norm3 = LayerNorm(channels)
        self.ffn = FFN(channels, expansion_factor, drop_out_rate)

    def forward(self, inp):
        x, condition = inp
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(condition), x)
        x = x + self.ffn(self.norm3(x))
        return x, condition

