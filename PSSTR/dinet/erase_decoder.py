import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from dinet.common.down_up_sample import UpSample
from dinet.decoders.edge.edge_fusion import FusionModule, Norm2d
from dinet.blocks.swinv2_block import SwinTransformerV2Decoder
import torch.nn.init as init

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class EraseDecoder(nn.Module):
    def __init__(self,
                 Block,
                 InterBlock,
                 SegAux,
                 RMAux,
                 dec_num_block=[1, 1, 1, 1],
                 channels=[32, 64, 128, 256],
                 input_size=256,
                 up_4x=True,
                 output_sizes=None
                 ):
        super().__init__()

        # channels: [64, 128, 256] ==> [256, 128, 64]
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

        # [256, 128, 64, 32]
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

        # [256, 128, 64]
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
            [UpSample(channel) for channel in list((channels))]
        )

        # self.reduces_rm = nn.ModuleList(
        #     [
        #         nn.Conv2d(channels[i] * 2, channels[i], kernel_size=1, bias=True)
        #         for i in range(len(channels))
        #     ]
        # )
        # self.reduces_seg = nn.ModuleList(
        #     [
        #         nn.Conv2d(channels[i] * 2, channels[i], kernel_size=1, bias=True)
        #         for i in range(len(channels))
        #     ]
        # )
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

        dim = channels[-1]
        self.out_rm_high_res = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, 3, kernel_size=3, padding=1)
        )

        self.out_seg_high_res = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=3, padding=1)
        )

        self.input_size = input_size
        self.match_output_size = False
        self.match_output_edge = None
        self.match_output_seg = None
        self.match_output_rm = None
        if not output_sizes is None and output_sizes[0] != input_size:
            self.match_output_size = True
            self.out_edge.pop(-1)
            self.out_seg.pop(-1)
            self.out_rm.pop(-1)
            op_size = output_sizes[0] # 最后一层的输出size
            depths = [2, 2, 2, 2]
            dep = []
            i = 0
            while op_size != input_size:
                dep.append(depths[i])
                op_size = op_size * 2
                i += 1
            ch = channels[-1] // (2 ** i)
            self.match_output_edge = SwinTransformerV2Decoder(embed_dim=channels[-1],window_size=8,depths=dep)
            self.match_output_seg = SwinTransformerV2Decoder(embed_dim=channels[-1],window_size=8,depths=dep)
            self.match_output_rm = SwinTransformerV2Decoder(embed_dim=channels[-1],window_size=8,depths=dep)
            self.out_edge.append(nn.Conv2d(ch, 1, kernel_size=3, padding=1, bias=True))
            self.out_seg.append(nn.Conv2d(ch, 1, kernel_size=3, padding=1, bias=True))
            self.out_rm.append(nn.Conv2d(ch, 3, kernel_size=3, padding=1, bias=True))


    def forward(self, image_embeddings, mask_embeddings):
        # if len(encs[0].shape) == 3:
        #     for i in range(len(encs)):
        #         h = int(math.sqrt(encs[i].shape[1]))
        #         encs[i] = rearrange(encs[i], 'b (h w) c -> b c h w', h=h, w=h)
        erase_feat = self.decoders_rm_init(image_embeddings)
        masks_feat = self.decoders_seg_init(mask_embeddings)

        # edge_feats = []
        # for i in range(len(self.get_edge_feat)):
        #     edge_feat = F.interpolate(self.get_edge_feat[i](encs[-1]), size=encs[i].size()[2:], mode='bilinear',
        #                               align_corners=True)
        #     edge_feats.append(edge_feat)

        out_edges, out_segs, out_rms = [], [], []

        for (
                decoder_seg,
                decoder_rm,
                up_seg,
                up_rm,
                # reduce_seg,
                # reduce_rm,
                seg_aux,
                rm_aux,
                # out_edge,
                out_seg,
                out_rm,
                # enc,
                # edge_feat,
                # seg_edge_fusion,
                step,
        ) in zip(
            self.decoders_seg,
            self.decoders_rm,
            self.ups_seg,
            self.ups_rm,
            # self.reduces_seg,
            # self.reduces_rm,
            self.seg_aux,
            self.rm_aux,
            # self.out_edge,
            self.out_seg,
            self.out_rm,
            # encs,
            # edge_feats,
            # self.seg_edge_fusion,
            range(self.layer_decoders)
        ):
            # seg, edge_feat = seg_edge_fusion(seg, edge_feat)
            masks_feat = decoder_seg(
                (
                    masks_feat,
                    # seg_aux(rm, enc, seg),
                    seg_aux(masks_feat, erase_feat)
                )
            )[0]
            erase_feat = decoder_rm(
                (
                    erase_feat,
                    # rm_aux(rm, enc, seg),
                    rm_aux(erase_feat, masks_feat),
                )
            )[0]

            # if step == self.layer_decoders - 1 and self.match_output_size: # deconv for last layer on 512
            #     rm = self.match_output_rm(rm)
            #     seg = self.match_output_seg(seg)
            #     edge_feat = self.match_output_edge(edge_feat)

            out_rms.append(out_rm(erase_feat))
            out_segs.append(out_seg(masks_feat))
            # out_edges.append(F.sigmoid(out_edge(edge_feat)))

            if step < self.layer_decoders - 1: # only upsample for [layer_decoders - 1] times
                masks_feat = up_seg(masks_feat)
                erase_feat = up_rm(erase_feat)
        
        out_rms.append(self.out_rm_high_res(erase_feat))
        out_segs.append(self.out_seg_high_res(masks_feat))

        # com_out = out_segs[-1] * out_rms[-1] + (1 - out_segs[-1]) * inp

        # return out_rms, out_segs, out_edges
        return out_rms, out_segs