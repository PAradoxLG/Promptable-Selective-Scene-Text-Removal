import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from models.model_util import create_backbone, create_decoder
from models.common.down_up_sample import DownSample, UpSample


class Network(nn.Module):  # all
    def __init__(
        self,
        backbone='swin_v2_tiny',
        decoder='parallel',
        Block='vit',
        InterBlock='vit',
        SegAux='abs_diff',
        RMAux='add',
        SegAuxList=[],
        RMAuxList=[],
        dec_num_block=[1, 1, 1, 1],
        aux_num_blocks=[2, 2, 2, 2],
        channels=[96, 192, 384, 768],
        input_size=256,
        add_region_swap=False,
        up_4x=True
    ):
        super().__init__()
        self.backbone_name = backbone
        encoder, _, channels = create_backbone(backbone, up_4x)
        print("encode_channels ", channels)
        dec_num_block = dec_num_block
        self.encoder = encoder
        # decoder
        if SegAuxList:
            SegAux = SegAuxList
        if RMAuxList:
            RMAux = RMAuxList
        print("SegAux: ", SegAux)
        print("RMAux: ", RMAux)
        self.decoder = create_decoder(decoder,
                                      Block, InterBlock, SegAux,
                                      RMAux, dec_num_block, aux_num_blocks, channels, add_region_swap, up_4x=up_4x, input_size=input_size)

        self.input_size = input_size

        # factor = 4
        # self.down_sample = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=True),
        #                                  nn.Conv2d(3, 3, kernel_size=3, padding=1),
        #                                  nn.Upsample(scale_factor=0.5, mode='bilinear'))

        # self.up_rm = nn.Sequential(nn.Conv2d(3, 3 * factor, kernel_size=3, padding=1, bias=True),
        #                             nn.PixelShuffle(2))
        # self.up_com = nn.Sequential(nn.Conv2d(3, 3 * factor, kernel_size=3, padding=1, bias=True),
        #                             nn.PixelShuffle(2))
        # self.up_seg = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
        #                             nn.Upsample(scale_factor=2, mode='bicubic'))
        # self.up_edge = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
        #                             nn.Upsample(scale_factor=2, mode='bicubic'))

    def forward(self, x):
        b, c, h, w = x.shape

        if 'resnet' in self.backbone_name:
            inp = x
            encs = self.encoder.forward_features(x)
            out_rms, com_out, out_segs, out_edges = self.decoder(inp, encs)

            # for i in range(len(out_rms)):
            #     out_rms[i] = F.interpolate(out_rms[i], tuple(dim * 2 for dim in out_rms[i].shape[2:]), mode='bilinear', align_corners=True)

            # com_out = F.interpolate(com_out, tuple(dim * 2 for dim in com_out.shape[2:]), mode='bilinear', align_corners=True)

            # for i in range(len(out_segs)):
            #     out_segs[i] = torch.clip(F.interpolate(out_segs[i], tuple(dim * 2 for dim in out_segs[i].shape[2:]), mode='bilinear', align_corners=True), 0.0, 1.0)

            # for i in range(len(out_rms)):
            #     out_edges[i] = torch.clip(F.interpolate(out_edges[i], tuple(dim * 2 for dim in out_edges[i].shape[2:]), mode='bilinear', align_corners=True), 0.0, 1.0)

            out_rms[-1] = self.up_rm(out_rms[-1])
            com_out = self.up_com(com_out)
            out_segs[-1] = F.sigmoid(self.up_seg(out_segs[-1]))
            out_edges[-1] = F.sigmoid(self.up_edge(out_edges[-1]))

        else:
            inp = x

            # if self.input_size == 512:
            #     x = self.down_sample(x)

            encs = self.encoder.forward_features(x)
            out_rms, com_out, out_segs, out_edges = self.decoder(inp, encs)

            # if h == 512 or w == 512:
            #     out_rms[-1] = self.up_rm(out_rms[-1])
            #     com_out = self.up_com(com_out)
            #     out_segs[-1] = F.sigmoid(self.up_seg(out_segs[-1]))
            #     out_edges[-1] = F.sigmoid(self.up_edge(out_edges[-1]))
            # com_out = torch.clip(com_out, 0., 1.)
            # for i in range(len(out)):
            #     out[i] = torch.clip(out[i], 0., 1.)
        return com_out, *out_rms, *out_segs, *out_edges
