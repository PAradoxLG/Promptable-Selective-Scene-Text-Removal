import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from models.common.down_up_sample import UpSample

class ParallelDecoder(nn.Module):
    def __init__(self,
                 Block,
                 InterBlock,
                 dec_num_block=[1, 1, 1, 1],
                 channels=[96, 192, 384, 768],
                 up_4x=True
                 ):
        super().__init__()

        self.decoders_rm_init = nn.Sequential(
            Block(channels[-1], 32),
            Block(channels[-1], 32),
        )

        self.decoders_rm = nn.ModuleList(
            [
                nn.Sequential(*[InterBlock(channel, patch_size) for _ in range(block)])
                for block, channel, patch_size in list(zip(dec_num_block, channels, [256, 128, 64, 32]))[::-1][1:]
            ]
        )

        self.decoders_seg_init = nn.Sequential(
            Block(channels[-1], 32),
            Block(channels[-1], 32),
        )

        self.decoders_seg = nn.ModuleList(
            [
                nn.Sequential(*[InterBlock(channel, patch_size) for _ in range(block)])
                for block, channel, patch_size in list(zip(dec_num_block, channels, [256, 128, 64, 32]))[::-1][1:]
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
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=True)
                for i in reversed(range(len(channels)))
            ]
        )
        self.reduces_seg = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=True)
                for i in reversed(range(len(channels)))
            ]
        )

        self.out_seg = nn.ModuleList(
            [
                nn.Conv2d(channels[i], 1, kernel_size=3, padding=1, bias=True)
                for i in reversed(range(len(channels)))
            ]
        )
        self.out_rm = nn.ModuleList(
            [
                nn.Conv2d(channels[i], 3, kernel_size=3, padding=1, bias=True)
                for i in reversed(range(len(channels)))
            ]
        )

    def forward(self, inp, encs):
        if len(encs[0].shape) == 3:
            for i in range(len(encs)):
                h = int(math.sqrt(encs[i].shape[1]))
                encs[i] = rearrange(encs[i], 'b (h w) c -> b c h w', h=h, w=h)
        seg = self.decoders_seg_init(encs[-1])
        rm = self.decoders_rm_init(encs[-1])
        encs = encs[:-1]
        out_rms = [self.out_rm[0](rm)]
        out_segs = [F.sigmoid(self.out_seg[0](seg))]
        for (
                decoder_seg,
                decoder_rm,
                up_seg,
                up_rm,
                reduce_seg,
                reduce_rm,
                out_seg,
                out_rm,
                enc,
        ) in zip(
            self.decoders_seg,
            self.decoders_rm,
            self.ups_seg,
            self.ups_rm,
            self.reduces_seg,
            self.reduces_rm,
            self.out_seg[1:],
            self.out_rm[1:],
            encs[::-1],
        ):
            seg = up_seg(seg)
            rm = up_rm(rm)
            seg = decoder_seg(
                (
                    reduce_seg(torch.cat([seg, enc], dim=1)),
                    reduce_seg(torch.cat([seg, enc], dim=1)),
                )
            )[0]
            rm = decoder_rm(
                (
                    reduce_rm(torch.cat([rm, enc], dim=1)),
                    reduce_rm(torch.cat([rm, enc], dim=1)),
                )
            )[0]

            out_rms.append(out_rm(rm))
            out_segs.append(F.sigmoid(out_seg(seg)))

        com_out = out_segs[-1] * out_rms[-1] + (1 - out_segs[-1]) * inp
        return out_rms, com_out, out_segs