import torch.nn as nn
import torch
# from models.common.down_up_sample import DownSample

class MSA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, 1)

        self.scale_14 = nn.Sequential(
            nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, groups=dim//4),
            nn.Conv2d(dim//4, dim//4, kernel_size=7, padding=6, groups=dim//4, dilation=2),
        )

        self.scale_21 = nn.Sequential(
            nn.Conv2d(dim//4, dim//4, kernel_size=5, padding=2, groups=dim//4),
            nn.Conv2d(dim//4, dim//4, kernel_size=7, padding=9, groups=dim//4, dilation=3),
        )

        self.scale_28 = nn.Sequential(
            nn.Conv2d(dim//4, dim//4, kernel_size=7, padding=3, groups=dim//4),
            nn.Conv2d(dim//4, dim//4, kernel_size=7, padding=12, groups=dim//4, dilation=4),
        )

        self.scale_35 = nn.Sequential(
            nn.Conv2d(dim//4, dim//4, kernel_size=9, padding=4, groups=dim//4),
            nn.Conv2d(dim//4, dim//4, kernel_size=7, padding=15, groups=dim//4, dilation=5),
        )

        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.conv(x).chunk(4, dim=1)

        attn1 = self.scale_14(x1)
        attn2 = self.scale_21(x2)
        attn3 = self.scale_28(x3)
        attn4 = self.scale_35(x4)
        attn = torch.cat([attn1, attn2, attn3, attn4], dim=1)
        x = x * attn
        attn = self.channel(x)
        return attn * x


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.net(x)

class MSACN_CS_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return x+self.net(x)


class Cross_MSACN_CS_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.reduce = nn.Conv2d(channels*2, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, inp):
        if isinstance(inp, tuple):
            x, condition = inp
            tmp = self.reduce(torch.cat([x, condition], dim=1))
            x = x + self.net(self.norm1(tmp))
            return x, condition
        else:
            x = inp
            return x + self.net(x)
        
def main():
    channel = 64
    size = 256
    model = Cross_MSACN_CS_Block(channel)
    total = sum(p.numel() for p in model.parameters())
    print(total/1e6, "M")

    import time
    x = torch.randn(16, channel, size, size)
    y = x.clone().detach()
    start = time.time()
    res = model([x, y])
    end = time.time()
    print(end - start)
    # print(res.shape)

if __name__ == '__main__':
    main()