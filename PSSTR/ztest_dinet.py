import argparse
from dinet.model_util import create_decoder
import torch

def test():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)
    parser.add_argument('--decoder', type=str, default='dualhelix')
    parser.add_argument('--Block', type=str, default='cnn')
    parser.add_argument('--InterBlock', type=str, default='cnn')
    parser.add_argument('--SegAux', type=str, default='abs_diff')
    parser.add_argument('--SegAuxList', nargs='+')
    parser.add_argument('--RMAux', type=str, default='abs_diff')
    parser.add_argument('--RMAuxList', nargs='+')
    parser.add_argument('--dec_num_block', nargs='+',type=int, default=[1, 2, 2])
    parser.add_argument('--aux_num_blocks', nargs='+',type=int, default=[2, 2, 2])
    parser.add_argument('--channels', nargs='+',type=int, default=[64, 128, 256])

    args = parser.parse_args()

    erase_decoder = create_decoder(args.decoder, args.Block, args.InterBlock,
                                   args.SegAux, args.RMAux,
                                   args.dec_num_block, args.aux_num_blocks,
                                   args.channels)
    
    x = torch.randn(1, 256, 64, 64)
    y = torch.randn(1, 256, 64, 64)
    erase, masks = erase_decoder(x, y)

    for (op1, m1) in zip(erase, masks):
        print(op1.shape, m1.shape)


    from thop import profile
    flops, params = profile(erase_decoder, inputs=(x,y))
    print(f"FLOPs: {flops/1e9:.2f}G")
    print(f"Parameters: {params/1e6:.2f}M")

def main():
    test()


if __name__ == '__main__':
    main()