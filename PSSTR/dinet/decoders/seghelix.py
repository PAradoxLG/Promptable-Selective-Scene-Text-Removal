import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from models.common.down_up_sample import UpSample

class SegGuidanceDecoder(nn.Module):
    def __init__(self,
                 Block,
                 InterBlock,
                 SegAux,
                 dec_num_block=[1, 1, 1, 1],
                 channels=[96, 192, 384, 768],
                 up_4x=True
                 ):
        super().__init__()
        self.decoders_rm_init = nn.Sequential(
            Block(channels[-1]),
            Block(channels[-1])
        )

        self.decoders_rm = nn.ModuleList(
            [
                nn.Sequential(*[InterBlock(channel) for _ in range(block)])
                for block, channel in list(zip(dec_num_block, channels))[::-1][1:]
            ]
        )

        self.decoders_seg_init = Block(channels[-1])

        self.decoders_seg = nn.ModuleList(
            [
                nn.Sequential(*[InterBlock(channel) for _ in range(block)])
                for block, channel in list(zip(dec_num_block, channels))[::-1][1:]
            ]
        )

        self.ups_rm = nn.ModuleList(
            [UpSample(channel) for channel in list(reversed(channels))[:-1]]
        )
        self.ups_seg = nn.ModuleList(
            [UpSample(channel, mode='bilinear') for channel in list(reversed(channels))[:-1]]
        )

        self.reduces_rm = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(range(len(channels)))
            ]
        )
        self.reduces_seg = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(range(len(channels)))
            ]
        )
        self.seg_aux = SegAux

        self.seg = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.seg_unembedd = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] * 4, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(channels[0], channels[0] * 4, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.PixelShuffle(2),
        )

        self.rm = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1, bias=False)
        self.rm_unembedd = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] * 4, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(channels[0], channels[0] * 4, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.PixelShuffle(2),
        )

    def forward(self, inp, encs):
        if len(encs[0].shape) == 3:
            for i in range(len(encs)):
                h = int(math.sqrt(encs[i].shape[1]))
                encs[i] = rearrange(encs[i], 'b (h w) c -> b c h w', h=h, w=h)
        seg = self.decoders_seg_init(encs[-1])
        rm = self.decoders_rm_init(encs[-1])
        encs = encs[:-1]

        for (
                decoder_seg,
                decoder_rm,
                up_seg,
                up_rm,
                reduce_seg,
                reduce_rm,
                seg_aux,
                rm_aux,
                enc,
        ) in zip(
            self.decoders_seg,
            self.decoders_rm,
            self.ups_seg,
            self.ups_rm,
            self.reduces_seg,
            self.reduces_rm,
            self.seg_aux,
            self.rm_aux,
            encs[::-1],
        ):
            seg = up_seg(seg)
            rm = up_rm(rm)
            seg = decoder_seg(
                (
                    reduce_seg(torch.cat([seg, enc], dim=1)),
                    seg_aux(torch.cat([seg, enc], dim=1)),
                )
            )[0]
            rm = decoder_rm(
                (
                    reduce_rm(torch.cat([rm, enc], dim=1)),
                    reduce_rm(torch.cat([rm, enc], dim=1)),
                )
            )[0]

        rm = self.rm(self.rm_unembedd(rm))
        seg = self.seg(self.seg_unembedd(seg))

        com_out = seg * rm + (1 - seg) * inp
        com_out = com_out.contiguous()

        return rm, com_out, seg