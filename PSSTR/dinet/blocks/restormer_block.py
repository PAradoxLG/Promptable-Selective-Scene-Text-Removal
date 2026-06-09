import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from dinet.common.norm import LayerNorm

class FFN(nn.Module):
    def __init__(self, dim, expansion_factor=1):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1,
                              groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim * 3,
                dim * 3,
                kernel_size=3,
                padding=1,
                groups=dim * 3,
                bias=False,
            ),
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class CrossMDTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, q, kv):
        b, c, h, w = q.shape
        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, expansion_factor=1):
        super().__init__()

        self.norm1 = LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm(channels)
        self.ffn = FFN(channels, expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlockWithCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=4, expansion_factor=1):
        super().__init__()

        self.norm1 = LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm(channels)
        self.cross_attn = CrossMDTA(channels, num_heads)
        self.norm3 = LayerNorm(channels)
        self.ffn = FFN(channels, expansion_factor)

    def forward(self, inp):
        x, condition = inp
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(condition), x)
        x = x + self.ffn(self.norm3(x))
        return x, condition
