import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import math
from models.model_util import create_backbone, create_decoder


class Network(nn.Module):  # all
    def __init__(
        self,
        backbone='swin_v2_tiny',
        decoder='parallel',
        Block='vit',
        InterBlock='vit',
        SegAux='abs_diff',
        RMAux='add',
        SegAuxList = [],
        RMAuxList = [],
        dec_num_block = [1, 1, 1, 1],
        aux_num_blocks=[2,2,2,2],
        channels=[96, 192, 384, 768],
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
        self.decoder=create_decoder(decoder,
                   Block, InterBlock, SegAux,
                   RMAux, dec_num_block, aux_num_blocks, channels,add_region_swap,up_4x=up_4x)

    def forward(self, x):
        inp = x
        encs = self.encoder.forward_features(x)
        # for i in range(len(encs)):
        #     print("encs[%d].shape = %s" % (i, encs[i].shape))
        out_rms, com_out, out_segs = self.decoder(inp, encs)
        # com_out = torch.clip(com_out, 0., 1.)
        # for i in range(len(out)):
        #     out[i] = torch.clip(out[i], 0., 1.)
        return com_out, *out_rms, *out_segs