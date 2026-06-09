"""This package contains modules related to function, network architectures, and models"""
import torch
import torch.nn as nn

from dinet.decoders.dualhelix import DualHelixDecoder
from dinet.erase_decoder import EraseDecoder
# from dinet.decoders.dualhelix_edge import DualHelixDecoder as DualHelixDecoder_with_edge
# from dinet.decoders.parallel import ParallelDecoder
# from dinet.decoders.dualhelix_fft import DualHelixDecoder as FFTDualHelixDecoder
# from dinet.decoders.parallel_fft import ParallelDecoder as FFTParallelDecoder
# from dinet.decoders.triple_fuse import TripleDecoder as TripleFuse
# from dinet.decoders.triple_parallel import TripleDecoder as Triple_parallel
# from dinet.decoders.triple_parallel_edge import TripleDecoder as Triple_parallel_edge
# from dinet.decoders.triple_fuse_edge import TripleDecoder as TripleFuse_edge
# from dinet.decoders.triple_fuse_edge_cross import TripleDecoder as TripleFuse_edge_cross
from dinet.interaction_mechanism.generate_inter_condition import add, abs_diff, diff, multiply, conv_1x1, ghost_dip, sim_se,interactive_fft_enhanced_attention,sim_ffta,sim_mdta,attn_inter,sim_ffn,cross_ffta,gate_correct
from dinet.blocks import restormer_block, vision_transformer_block, cnn_block

out_backbone = ''


def create_aux(aux,channels,aux_num_blocks,input_size=256):
    if aux == 'add':
        aux =  [add, add, add, add]
    elif aux == 'diff':
        aux =  [diff, diff, diff, diff]
    elif aux == 'abs_diff':
        aux = nn.ModuleList([abs_diff(), abs_diff(), abs_diff(), abs_diff()])
    elif aux == 'multiply':
        aux = [multiply, multiply, multiply, multiply]
    elif aux == 'conv_1x1':
        aux = nn.ModuleList(
            [
                conv_1x1(channels[i]) for i in list(range(len(channels)))[::-1][1:]
            ]
        )
    elif aux == 'ghost_dip':
        aux = nn.ModuleList(
            [
                ghost_dip(channels[i]) for i in list(range(len(channels)))[::-1][1:]
            ]
        )
    elif aux == 'interactive_fft_enhanced_attention':
        aux = nn.ModuleList(
            [
                interactive_fft_enhanced_attention(channels[i]) for i in list(range(len(channels)))[::-1][1:]
            ]
        )
    elif aux =='sim_ffta':
        patchin = 64
        patch_sizes = [ patchin * (2**i) for i in range(len(channels))]
        patch_sizes = patch_sizes[::-1]
        aux = nn.ModuleList(
            [
                sim_ffta(channels[i],patch_sizes[i],aux_num_blocks[i]) for i in list(range(len(channels)))[::-1]
            ]
        )
    elif aux =='sim_se':
        aux = nn.ModuleList(
            [
                sim_se(channels[i]) for i in list(range(len(channels)))[::-1][1:]
            ]
        )
    elif aux =='sim_mdta':
        aux = nn.ModuleList(
            [
                sim_mdta(channels[i]) for i in list(range(len(channels)))[::-1][1:]
            ]
        )
    elif aux =='attn_inter':
        aux = nn.ModuleList(
            [
                attn_inter(channels[i], aux_num_blocks[i]) for i in list(range(len(channels)))[::-1]
            ]
        )
    elif aux == 'triple_fuse':
        aux = nn.ModuleList(
            [
                triple_fuse(channels[i], channels[i]) for i in list(range(len(channels)))[::-1]
            ]
        )
    elif aux == 'sim_ffn':
        aux = nn.ModuleList(
            [
                sim_ffn(channels[i], aux_num_blocks[i]) for i in list(range(len(channels)))[::-1]
            ]
        )
    elif aux == 'gate_correct':
        ### lg
        aux = nn.ModuleList(
            [
                gate_correct(channels[i], aux_num_blocks[i]) for i in list(range(len(channels)))[::-1]
            ]
        )
    return aux

def create_block(block):
    if block == 'vit':
        block = vision_transformer_block.TransformerBlock
    elif block == 'restormer':
        block = restormer_block.TransformerBlock
    elif block == 'wave':
        block = wave_block.WaveTransformerBlock
    elif block == 'cnn':
        block = cnn_block.MSACN_CS_Block
    elif block == 'fft':
        block = fft_block.TransformerBlock
    elif block == 'fftstr':
        block = fftstr_block.TransformerBlock
    return block

def create_inter_block(block):
    if block == 'vit':
        block = vision_transformer_block.TransformerBlockWithCrossAttention
    elif block == 'restormer':
        block = restormer_block.TransformerBlockWithCrossAttention
    elif block == 'wave':
        block = wave_block.WaveTransformerBlockWithCrossAttention
    elif block == 'cnn':
        block = cnn_block.Cross_MSACN_CS_Block
    elif block == 'fft':
        block = fft_block.TransformerBlockWithCrossAttention
    elif block == 'fftstr':
        block = fftstr_block.TransformerBlockWithCrossAttention
    return block

def create_decoder(decoder,
                   Block, InterBlock, SegAux,
                   RMAux, dec_num_block, aux_num_blocks, channels,input_size=256):

    Block = create_block(Block)
    InterBlock = create_inter_block(InterBlock)
    if SegAux is not None:
        SegAux = create_aux(SegAux,channels,aux_num_blocks,input_size)

    if RMAux is not None:
        RMAux = create_aux(RMAux,channels,aux_num_blocks,input_size)
   
    if decoder == 'parallel':
        decoder = ParallelDecoder(
            Block,
            InterBlock,
            dec_num_block,
            channels,
        )
    elif decoder == 'dualhelix':
        decoder = EraseDecoder(
            Block,
            InterBlock,
            SegAux,
            RMAux,
            dec_num_block,
            channels,
        )
    elif decoder == 'dualhelix_edge':
        decoder = DualHelixDecoder_with_edge(
            Block,
            InterBlock,
            SegAux,
            RMAux,
            dec_num_block,
            channels,
        )
    elif decoder == 'triple':
        decoder = TripleFuse(
            Block,
            InterBlock,
            SegAux,
            RMAux,
            dec_num_block,
            channels,
        )
    elif decoder == 'triple_edge':
        decoder = TripleFuse_edge(
            Block,
            InterBlock,
            SegAux,
            RMAux,
            dec_num_block,
            channels,
            input_size=input_size,
        )
    elif decoder == 'triple_edge_cross':
        decoder = TripleFuse_edge_cross(
            Block,
            InterBlock,
            SegAux,
            RMAux,
            dec_num_block,
            channels,
            input_size=input_size,
        )
    elif decoder == 'triple_parallel':
        decoder = Triple_parallel(
            Block,
            InterBlock,
            dec_num_block,
            channels,
            input_size=input_size,
        )
    elif decoder == 'triple_parallel_edge':
        decoder = Triple_parallel_edge(
            Block,
            InterBlock,
            dec_num_block,
            channels,
        )
    return decoder