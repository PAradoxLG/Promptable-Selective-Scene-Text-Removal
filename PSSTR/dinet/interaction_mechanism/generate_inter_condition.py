import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from dinet.common.norm import LayerNorm, FFTLayerNorm

# no parametter
class abs_diff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.abs(x1 - x2)

# def abs_diff(x1,x2):
#     return torch.abs(x1-x2)

def diff(x1,x2):
    return x1-x2

class add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

def multiply(x1,x2):
    return x1*x2

# with parametter

class FFTA(nn.Module):
    def __init__(self, dim,patch_size=8):
        super().__init__()

        self.patch_size = patch_size

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.proj_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),
                                      nn.ReLU(inplace=True))

        self.norm = FFTLayerNorm(dim, LayerNorm_type='WithBias')

    def forward(self, q, k, v):

        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.proj_out(output)
        return output


class CrossFFTA(nn.Module):
    def __init__(self, dim,patch_size=8):
        super().__init__()

        # self.qc = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # self.kc = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.vc = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.patch_size = patch_size // 4
        self.weight1 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.weight2 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))

        self.norm = FFTLayerNorm(dim, LayerNorm_type='WithBias')

        self.proj_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                                    nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        # q = self.qc(x1)
        # k = self.kc(x2)
        # v = self.vc(x2)

        q = x1
        k = x2
        v = x2

        q = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        
        q_fft = torch.fft.rfft2(q, norm='ortho') * self.weight1
        k_fft = torch.fft.rfft2(k, norm='ortho') * self.weight2

        attn = q_fft * k_fft
        attn = torch.fft.irfft2(attn, s=(self.patch_size,self.patch_size), norm='ortho')

        attn = rearrange(attn, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                                patch2=self.patch_size)

        output = attn * v
        
        return self.proj_out(output)


class cross_ffta(nn.Module):
    def __init__(self, channels, patch_size=16, blocks=2):
        super().__init__()

        # interactive module
        self.qc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.kvc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.blocks = blocks
        self.attn_body = nn.ModuleList([CrossFFTA(channels, patch_size) for _ in range(blocks)])
            
    def forward(self, x, condition):
        x = self.qc(x)
        condition = self.kvc(condition)
        for i in range(self.blocks):
            attn = self.attn_body[i]
            x = x + attn(x, condition)

        return x


class sim_ffta(nn.Module):
    def __init__(self, channels, patch_size=16, blocks=2):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )
        # self.body = nn.Sequential(
        #     *[FFTA(channels, patch_size) for _ in range(blocks)]
        # )
        self.body = nn.ModuleList(
            [FFTA(channels, patch_size) for _ in range(blocks)]
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2):
        # x = self.proj_in(torch.cat([x1,x2],dim=1))
        for layer in self.body:
            attn = layer(x1, x2, x2)
            x1 = x1 + attn
        return x1 + self.proj_out(x1)



class sim_ffta512(nn.Module):
    def __init__(self, channels, patch_size=16, blocks=2):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
        )
        self.body = nn.Sequential(
            *[FFTA512(channels, patch_size) for _ in range(blocks)]
        )

    def forward(self, x1, x2):
        x = self.proj_in(torch.cat([x1,x2],dim=1))
        x = self.body(x)
        return x    

class FFTA512(nn.Module):
    def __init__(self, dim,patch_size=8):
        super().__init__()

        self.patch_size = patch_size // 16
        self.weight1 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.weight2 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        self.weight3 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))

        self.conv1d = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                                    nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x):

        f = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        f = torch.fft.rfft2(f.float(), norm='ortho')
        q_fft = f * self.weight1
        k_fft = f * self.weight2
        v_fft = f * self.weight3
        out = q_fft * k_fft * v_fft
        out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size), norm='ortho')

        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        return x+self.conv1d(out)

class triple_fuse(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(triple_fuse, self).__init__()

        self.conv = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=3, padding=1, bias=False)                  
                                )

        
    def forward(self, rm, enc, seg):
        seg = torch.sigmoid(seg)
        return self.conv(seg*rm + (1-seg)*enc)

class attn_inter(nn.Module):
    def __init__(self, channels, blocks=2):
        super().__init__()

        self.body = nn.ModuleList([Attention(channels) for _ in range(blocks)])
        self.proj_in = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

    def forward(self, rm, enc, seg):
        x1 = seg
        x2 = torch.abs(enc - rm)
        for layer in self.body:
            pid_info = layer(x1, x2)
            x1 = self.proj_in(torch.cat([x1, pid_info],dim=1))
        return x1
    
class attn_inter2(nn.Module):
    def __init__(self, channels, blocks=2):
        super().__init__()

        self.body = nn.ModuleList([Attention(channels) for _ in range(blocks)])
        self.proj_in = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

    def forward(self, x1, x2):
        for layer in self.body:
            attn_info = layer(x1, x2)
            x1 = self.proj_in(torch.cat([x1, attn_info],dim=1))
        return x1

class Attention(nn.Module):
    def __init__(self, channels, num_heads=8, window_size=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        x1 = x1.view(b, -1, c)
        x2 = x2.view(b, -1, c)
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)

        q = rearrange(q, 'b hw (head c) -> b head hw c', head=self.num_heads)
        k = rearrange(k, 'b hw (head c) -> b head hw c', head=self.num_heads)
        v = rearrange(v, 'b hw (head c) -> b head hw c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out

class sim_ffn(nn.Module):
    def __init__(self, channels, blocks=1):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
        )
        self.body = nn.Sequential(
            *[FFN(channels) for _ in range(blocks)]
        )

    def forward(self, x1,x2):
        x = self.proj_in(torch.cat([x1,x2],dim=1))
        return x + self.body(x)

### lg
class gate_correct(nn.Module):
    def __init__(self, channels, blocks=1):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.body = nn.Sequential(
            *[FFN(channels) for _ in range(blocks)]
        )

    def forward(self, x1,x2):
        # x = self.proj_in(torch.cat([x1,x2],dim=1))
        # return x + self.body(x)
        weights = nn.functional.adaptive_avg_pool2d(x2, (1, 1))
        weights = torch.sigmoid(weights)
        x=x1+self.proj_in(x1*weights)
        return x + self.body(x) 

# 底下大多是测试 没用到
# class Cross_FFTA(nn.Module):
#     def __init__(self, dim, patch_size=8):
#         super().__init__()

#         self.patch_size = patch_size
#         self.ratio = 0.5
#         reduced_dim = int(dim * self.ratio)

#         self.weight1 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
#         self.weight2 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
#         self.weight3 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
        
#         self.conv1 = nn.Sequential(nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=True),
#                                             nn.BatchNorm2d(reduced_dim), nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=True),
#                                     nn.BatchNorm2d(reduced_dim), nn.ReLU(inplace=True))
        
#         self.conv3 = nn.Sequential(nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=True),
#                                     nn.BatchNorm2d(reduced_dim), nn.ReLU(inplace=True))
        
#         self.convt = nn.Sequential(nn.Conv2d(reduced_dim, reduced_dim, kernel_size=1, bias=True),
#                                     nn.BatchNorm2d(reduced_dim), nn.ReLU(inplace=True))

#         self.conv_out = nn.Sequential(nn.Conv2d(reduced_dim, dim, kernel_size=1, bias=True),
#                                     nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

#         self.dim = dim

#     def forward(self, x1, x2):
#         # x1: rm/seg  x2: enc
# # ours
#         # q = self.conv1(x2)
#         # k = self.conv2(x1)
#         # v = self.conv3(x1)
#         # # q_patch = rearrange(x2, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#         # #                     patch2=self.patch_size)

#         # # k_patch = rearrange(x1, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#         # #                     patch2=self.patch_size)
        
#         # # v_patch = rearrange(x1, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#         # #                     patch2=self.patch_size)
        
#         # q_fft = torch.fft.rfft(q.float(), norm='ortho') * self.weight1
#         # k_fft = torch.fft.rfft(k.float(), norm='ortho') * self.weight2
#         # v_fft = torch.fft.rfft(v.float(), norm='ortho') * self.weight3

#         # out = q_fft * k_fft * v_fft

#         # out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size), norm='ortho')

        
#         # # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
#         # #                 patch2=self.patch_size)

#  # FSAS

#         q = self.conv1(x2)
#         k = self.conv2(x1)
#         v = self.conv3(x1)
#         # q_patch = rearrange(x2, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#         #                     patch2=self.patch_size)

#         # k_patch = rearrange(x1, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#         #                     patch2=self.patch_size)
        
#         # v_patch = rearrange(x1, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#         #                     patch2=self.patch_size)
        
#         q_fft = torch.fft.rfft(q.float(), norm='ortho') * self.weight1
#         k_fft = torch.fft.rfft(k.float(), norm='ortho') * self.weight2

#         out = q_fft * k_fft

#         out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size), norm='ortho')

#         out = self.convt(out) * v

#         # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
#         #                 patch2=self.patch_size)
#         return self.conv_out(out)

# class cross_ffta(nn.Module):
#     def __init__(self, channels, patch_size=16, blocks=2):
#         super().__init__()

#         self.body = nn.ModuleList(
#             [Cross_FFTA(channels, patch_size) for _ in range(blocks)]
#         )

#     def forward(self, x1, x2):
#         # x1: rm/seg  x2: enc
#         for layer in self.body:
#             x1 = layer(x1, x2)
#         return x1

def fft_sa(x1,x2):
    h,w = x1.shape[2:]
    q = torch.fft.rfft2(x1, norm='ortho')
    kv = torch.fft.rfft2(x2, norm='ortho')
    out = q * kv * kv
    out = torch.fft.irfft2(out, s=(h,w), norm='ortho')

    # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
    #                 patch2=self.patch_size)
    return out

# with parametter
class conv_1x1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Conv2d(channels*2, channels, kernel_size=1, padding=0)

    def forward(self, x1,x2):
        x = torch.cat([x1, x2], dim=1)
        return self.net(x)


# class FFTA(nn.Module):
#     def __init__(self, channels, patch_size=8):
#         super().__init__()

#         self.patch_size = patch_size
#         self.weight1 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
#         self.weight2 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
#         self.weight3 = nn.Parameter(torch.zeros((1, self.patch_size, self.patch_size // 2 + 1)))
#         self.channel_mix = nn.Conv2d(channels*2, channels*2, kernel_size=1, bias=True)

#     def forward(self, x):
#         # 太慢了 不太行

#         # f = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#         #                     patch2=self.patch_size)
#         f = torch.fft.rfft2(x.float(), norm='ortho')
#         q_fft = f * self.weight1
#         k_fft = f * self.weight2
#         v_fft = f * self.weight3
#         out = q_fft * k_fft * v_fft
#         out_real = out.real
#         out_imag = out.imag
#         out = torch.cat([out_real, out_imag], 1)
#         out = self.channel_mix(out)
#         out_real, out_imag = out.chunk(2, dim=1)
#         out = torch.stack((out_real, out_imag), -1)
#         out = torch.view_as_complex(out)
#         out = torch.fft.irfft2(out, s=(self.patch_size,self.patch_size), norm='ortho')
#         # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
#         #                 patch2=self.patch_size)
#         return out

# class sim_ffta(nn.Module):
#     def __init__(self, channels, patch_size=16):
#         super().__init__()
#
#         self.proj_in = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
#         )
#         self.body = nn.Sequential(
#             FFTA(channels, patch_size),
#             FFTA(channels, patch_size),
#             # FFTA(channels, patch_size),
#         )
#
#     def forward(self, x1,x2):
#         x = self.proj_in(torch.cat([x1,x2],dim=1))
#         x = self.body(x)
#         return x

# class sim_ffta(nn.Module):
#     def __init__(self, channels, patch_size=16):
#         super().__init__()
#
#         self.proj_in = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
#         )
#         self.body = nn.Sequential(
#             FFTA(channels, patch_size),
#             FFTA(channels, patch_size),
#         )
#
#     def forward(self, x1, x2):
#         x = self.proj_in(torch.cat([x1, x2], dim=1))
#         x = self.body(x)
#         return x

# class sim_ffta(nn.Module):
#     def __init__(self, channels, patch_size=16, blocks=2):
#         super().__init__()

#         self.proj_in = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
#         )
#         self.body = nn.Sequential(
#             *[FFTA(channels, patch_size) for _ in range(blocks)]
#         )

#     def forward(self, x1,x2):
#         x = self.proj_in(torch.cat([x1,x2],dim=1))
#         x = self.body(x)
#         return x


class simpled_SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x+x*self.sca(x)

class sim_se(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.proj_in = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.body = nn.Sequential(
            simpled_SE(channels),
            simpled_SE(channels),
        )

    def forward(self, x1, x2):
        x = self.proj_in(torch.cat([x1, x2], dim=1))
        x = self.body(x)
        return x


class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
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

class sim_mdta(nn.Module):
    def __init__(self, channels, blocks=1):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
        )
        self.body = nn.Sequential(
            *[MDTA(channels) for _ in range(blocks)]
        )

    def forward(self, x1,x2):
        x = self.proj_in(torch.cat([x1,x2],dim=1))
        x = self.body(x)
        return x



class Resblock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels*2, channels, kernel_size=1, padding=0, bias=False)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x1,x2):
        x = self.reduce(torch.cat([x1, x2], dim=1))
        return x+self.reduce(x)

class SEblock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)

class SEResblock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels*2, channels, kernel_size=1, padding=0, bias=True)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            SEAttention(channels)
        )

    def forward(self, x1,x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.reduce(x)
        return x + self.body(x)

class simse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels*2, channels, kernel_size=1, padding=0, bias=True)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x1,x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.reduce(x)
        return x + x*self.body(x)









class conv_3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Conv2d(channels, channels//2, kernel_size=3, padding=1, bias=False)

    def forward(self, x1,x2):
        x = torch.cat([x1, x2], dim=1)
        return self.net(x)

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x



class simse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(channels//8, channels, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1,x2):
        x = torch.abs(x1-x2)
        x = x+x*self.sca(x)
        x = x + self.ffn(x)
        return x


class ghost_dip_attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.split_channels = channels//4
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)

        self.sobel_x = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
        self.laplace = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
        self.conv = nn.Conv2d(channels//4, channels//4, kernel_size=3, padding=1, groups=self.split_channels, bias=False)


        self.norm1 = LayerNorm(channels)
        self.norm2 = LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        x1,x2,x3,x4 = self.norm1(self.proj(x)).chunk(4, dim=1)
        x1 = F.conv2d(x1, self.sobel_x.repeat(self.split_channels, self.split_channels, 1,1), stride=1, padding=1)
        x2 = F.conv2d(x2, self.sobel_y.repeat(self.split_channels, self.split_channels,1,1), stride=1, padding=1)
        x3 = F.conv2d(x3, self.laplace.repeat(self.split_channels, self.split_channels,1,1), stride=1, padding=1)
        x4 = self.conv(x4)
        x = x + torch.cat([x1,x2,x3,x4],dim=1)*x
        x = x + self.ffn(self.norm2(x))
        return x

class ghost_dip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ghost_dip = ghost_dip_attention(channels)
    def forward(self, x1,x2):
        x1 = self.ghost_dip(x1)
        x2 = self.ghost_dip(x2)
        return torch.abs(x1-x2)


import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from dinet.common.norm import LayerNorm



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
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, kernel_size=1, padding=0, bias=True),
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


class FFTTransformerBlock(nn.Module):
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


class interactive_fft_enhanced_attention(nn.Module):
    def __init__(self, channels, patch_size=8):
        super().__init__()

        self.proj_in = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)
        self.body = nn.Sequential(
            FFTTransformerBlock(channels,patch_size),
            FFTTransformerBlock(channels,patch_size),
            FFTTransformerBlock(channels, patch_size),
            FFTTransformerBlock(channels, patch_size)
        )

    def forward(self, x1,x2):
        x = self.proj_in(torch.cat([x1,x2],dim=1))
        x = self.body(x)
        return x



# class sim_ffta(nn.Module):
#     def __init__(self, channels, patch_size=16):
#         super().__init__()
#
#         self.proj_in = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
#         )
#         self.body = nn.Sequential(
#             FFTA(channels, patch_size),
#             FFTA(channels, patch_size),
#             # FFTA(channels, patch_size),
#         )
#
#     def forward(self, x1,x2):
#         x = self.proj_in(torch.cat([x1,x2],dim=1))
#         x = self.body(x)
#         return x

# class sim_ffta(nn.Module):
#     def __init__(self, channels, patch_size=16):
#         super().__init__()
#
#         self.proj_in = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
#         )
#         self.body = nn.Sequential(
#             FFTA(channels, patch_size),
#             FFTA(channels, patch_size),
#         )
#
#     def forward(self, x1, x2):
#         x = self.proj_in(torch.cat([x1, x2], dim=1))
#         x = self.body(x)
#         return x

class simpled_SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x+x*self.sca(x)

class sim_se(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.proj_in = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.body = nn.Sequential(
            simpled_SE(channels),
            simpled_SE(channels),
        )

    def forward(self, x1, x2):
        x = self.proj_in(torch.cat([x1, x2], dim=1))
        x = self.body(x)
        return x


class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
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

class sim_mdta(nn.Module):
    def __init__(self, channels, blocks=1):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
        )
        self.body = nn.Sequential(
            *[MDTA(channels) for _ in range(blocks)]
        )

    def forward(self, x1,x2):
        x = self.proj_in(torch.cat([x1,x2],dim=1))
        x = self.body(x)
        return x