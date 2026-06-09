import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from models.common.down_up_sample import UpSample
from models.decoders.edge.edge_fusion import FusionModule, Norm2d

class TripleDecoder(nn.Module):
    def __init__(self,
                 Block,
                 InterBlock,
                 SegAux,
                 RMAux,
                 dec_num_block=[1, 1, 1, 1],
                 channels=[96, 192, 384, 768],
                 input_size=256,
                 up_4x=True
                 ):
        super().__init__()

        # channels: [48, 96, 192, 384] ==> [384, 192, 96, 48]
        channels = channels[::-1]
        self.layer_decoders = len(channels)

        # get edge feature and fuse [seg, edge] features
        self.get_edge_feat, self.seg_edge_fusion = [], []
        for i in range(len(channels)):
            self.get_edge_feat.append(nn.Sequential(nn.Conv2d(channels[-1], channels[i], kernel_size=3, padding=1, bias=False),
                                      Norm2d(channels[i]), nn.ReLU(inplace=True)))
            self.seg_edge_fusion.append(FusionModule(channels[i], channels[i]))

        self.get_edge_feat = nn.ModuleList(self.get_edge_feat)
        self.seg_edge_fusion = nn.ModuleList(self.seg_edge_fusion)

        # decoder interact from first layer
        self.decoders_rm_init = nn.Sequential(
            Block(channels[0]),
            Block(channels[0]),
        )

        # [384, 192, 96, 48]
        self.decoders_rm = nn.ModuleList(
            [
                nn.Sequential(*[InterBlock(channel) for _ in range(block)])
                for block, channel in list(zip(dec_num_block, channels))
            ]
        )

        self.decoders_seg_init = nn.Sequential(
            Block(channels[0]),
            Block(channels[0]),
        )

        # [384, 192, 96, 48]
        self.decoders_seg = nn.ModuleList(
            [
                nn.Sequential(*[InterBlock(channel) for _ in range(block)])
                for block, channel in list(zip(dec_num_block, channels))
            ]
        )

        self.ups_rm = nn.ModuleList(
            [UpSample(channel) for channel in list((channels))]
        )
        self.ups_seg = nn.ModuleList(
            [UpSample(channel, mode='bilinear') for channel in list((channels))]
        )

        #   (0): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1))
        #   (1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
        #   (2): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
        #   (3): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1))
        self.reduces_rm = nn.ModuleList(
            [
                nn.Conv2d(channels[i] * 2, channels[i], kernel_size=1, bias=True)
                for i in range(len(channels))
            ]
        )
        self.reduces_seg = nn.ModuleList(
            [
                nn.Conv2d(channels[i] * 2, channels[i], kernel_size=1, bias=True)
                for i in range(len(channels))
            ]
        )
        self.seg_aux = SegAux
        self.rm_aux = RMAux

        self.out_edge = nn.ModuleList(
            [
                nn.Conv2d(channels[i], 1, kernel_size=3, padding=1, bias=True)
                for i in range(len(channels))
            ]
        )
        self.out_seg = nn.ModuleList(
            [
                nn.Conv2d(channels[i], 1, kernel_size=3, padding=1, bias=True)
                for i in range(len(channels))
            ]
        )
        self.out_rm = nn.ModuleList(
            [
                nn.Conv2d(channels[i], 3, kernel_size=3, padding=1, bias=True)
                for i in range(len(channels))
            ]
        )

        self.input_size = input_size
        if self.input_size == 512:
            # deconv for rm, seg and edge result in last layer
            self.deconv_rm = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                           nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
                                           nn.BatchNorm2d(channels[-1]))
            self.deconv_seg = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                           nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
                                           nn.BatchNorm2d(channels[-1]))
            self.deconv_edge = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                           nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
                                           nn.BatchNorm2d(channels[-1]))

    def forward(self, inp, encs):
        if len(encs[0].shape) == 3:
            for i in range(len(encs)):
                h = int(math.sqrt(encs[i].shape[1]))
                encs[i] = rearrange(encs[i], 'b (h w) c -> b c h w', h=h, w=h)
        seg = self.decoders_seg_init(encs[-1])
        rm = self.decoders_rm_init(encs[-1])

        encs = encs[::-1] # encs = [384, 192, 96, 48]

        edge_feats = []
        for i in range(len(self.get_edge_feat)):
            edge_feat = F.interpolate(self.get_edge_feat[i](encs[-1]), size=encs[i].size()[2:], mode='bilinear',
                                      align_corners=True)
            edge_feats.append(edge_feat)

        out_edges, out_segs, out_rms = [], [], []

        for (
                decoder_seg,
                decoder_rm,
                up_seg,
                up_rm,
                reduce_seg,
                reduce_rm,
                seg_aux,
                rm_aux,
                out_edge,
                out_seg,
                out_rm,
                enc,
                edge_feat,
                seg_edge_fusion,
                step,
        ) in zip(
            self.decoders_seg,
            self.decoders_rm,
            self.ups_seg,
            self.ups_rm,
            self.reduces_seg,
            self.reduces_rm,
            self.seg_aux,
            self.rm_aux,
            self.out_edge,
            self.out_seg,
            self.out_rm,
            encs,
            edge_feats,
            self.seg_edge_fusion,
            range(self.layer_decoders)
        ):
            seg, edge_feat = seg_edge_fusion(seg, edge_feat)
            
            seg = decoder_seg(
                (
                    reduce_seg(torch.cat([seg, enc], dim=1)),
                    # seg_aux(rm, enc, seg),
                    seg_aux(rm, enc),
                )
            )[0]
            rm = decoder_rm(
                (
                    reduce_rm(torch.cat([rm, enc], dim=1)),
                    # rm_aux(rm, enc, seg),
                    rm_aux(seg, enc),
                )
            )[0]

            if step == self.layer_decoders - 1 and self.input_size == 512: # deconv for last layer on 512
                rm = self.deconv_rm(rm)
                seg = self.deconv_seg(seg)
                edge_feat = self.deconv_seg(edge_feat)

            out_rms.append(out_rm(rm))
            out_segs.append(F.sigmoid(out_seg(seg)))
            out_edges.append(F.sigmoid(out_edge(edge_feat)))

            if step < self.layer_decoders - 1: # only upsample for [layer_decoders - 1] times
                seg = up_seg(seg)
                rm = up_rm(rm)

        com_out = out_segs[-1] * out_rms[-1] + (1 - out_segs[-1]) * inp

        return out_rms, com_out, out_segs, out_edges