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
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.net(x)

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

class MDTA(nn.Module):
    def __init__(self, dim,patch_size=8):
        super().__init__()


        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True),
        )
        self.norm1 = LayerNorm(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.patch_size = patch_size
        self.weight = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))


    def forward(self, x):
        # 太慢了 不太行

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        b,c,h,w = q.shape
        q = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        v = rearrange(v, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        v_fft = torch.fft.rfft2(v.float())
        out = q_fft * k_fft * v_fft * self.weight
        out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size))

        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.project_out(self.norm1(out))

        return out


class CrossMDTA(nn.Module):
    def __init__(self, dim, patch_size=8):
        super().__init__()

        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.norm1 = LayerNorm(dim)
        self.patch_size = patch_size
        self.weight = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

    def forward(self, q, kv):
        # 太慢了 不太行

        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)
        b, c, h, w = q.shape

        q = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        v = rearrange(v, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        v_fft = torch.fft.rfft2(v.float())
        out = q_fft * k_fft * v_fft * self.weight
        out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size))

        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.project_out(self.norm1(out))

        return out

class TransformerBlock(nn.Module):
    def __init__(self, channels, patch_size=8,num_heads=4, expansion_factor=1,drop_out_rate=0.2):
        super().__init__()

        self.norm1 = LayerNorm(channels)
        self.attn = MDTA(channels, patch_size)
        self.norm2 = LayerNorm(channels)
        self.ffn = FFN(channels)
        self.drop1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.drop2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

class TransformerBlockWithCrossAttention(nn.Module):
    def __init__(self, channels,patch_size=8, num_heads=4, expansion_factor=1,drop_out_rate=0.2):
        super().__init__()

        self.norm1 = LayerNorm(channels)
        self.attn = MDTA(channels, patch_size)
        self.norm2 = LayerNorm(channels)
        self.cross_attn = CrossMDTA(channels, patch_size)
        self.norm3 = LayerNorm(channels)
        self.ffn = FFN(channels)
        self.drop1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.drop2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.drop3 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x, condition = inp
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.cross_attn(self.norm2(condition), x))
        x = x + self.drop3(self.ffn(self.norm3(x)))
        return x, condition
