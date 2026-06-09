import torch
from torch import nn
from torchinfo import summary

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = nn.BatchNorm2d
    normalization_layer = layer(in_channels)
    return normalization_layer

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.radio = 4

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // self.radio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // self.radio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MutualFusion(nn.Module):

    def __init__(self, inplane_x, inplane_b, norm_layer=Norm2d, dr2=2, dr4=4):
        super(MutualFusion, self).__init__()

        self.inplane_x = inplane_x # x
        self.inplane_b = inplane_b # y

        self.input_channels = inplane_x + inplane_b # x + y

        self.channels_single = int(self.input_channels / 4) # (x + y) / 4
        self.channels_double = int(self.input_channels / 2) # (x + y) / 2

        self.dr2 = dr2
        self.dr4 = dr4

        self.padding2 = 2 * dr2 # 4

        self.padding4 = 4 * dr4 # 16

        self.A = None

        self.p2_channel_reduction = nn.Sequential( # (x + y) -> (x + y) / 4
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p4_channel_reduction = nn.Sequential( # (x + y) -> (x + y) / 4
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p2_d1 = nn.Sequential( # (x + y) / 4 -> (x + y) / 4
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, padding=self.padding2,
                      dilation=self.dr2),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p2_fusion = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, dilation=1), # (x + y) / 4 -> (x + y) / 4
                                       norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p4_d1 = nn.Sequential( # (x + y) / 4 -> (x + y) / 4
            nn.Conv2d(self.channels_single, self.channels_single, 9, 1, padding=self.padding4,
                      dilation=self.dr4),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.p4_fusion = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, dilation=1), # (x + y) / 4 -> (x + y) / 4
                                       norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.channel_reduction = nn.Sequential( # (x + y) / 2 -> 2
            nn.Conv2d(in_channels=self.channels_double, out_channels=2, kernel_size=1),
        )

    def forward(self, x, x_b): # [seg, edge]
        concat_feature = torch.cat([x, x_b], dim=1) # x + y

        p2_input = self.p2_channel_reduction(concat_feature) # (x + y) -> (x + y) / 4
        p2 = self.p2_fusion(self.p2_d1(p2_input)) # (x + y) -> (x + y) / 4
        # print("p2.shape ", p2.shape)

        p4_input = self.p4_channel_reduction(concat_feature) # (x + y) -> (x + y) / 4
        p4 = self.p4_fusion(self.p4_d1(p4_input)) # (x + y) -> (x + y) / 4
        # print("p4.shape ", p4.shape)

        A = torch.sigmoid(self.channel_reduction(torch.cat((p2, p4), dim=1))) # (x + y) / 2 -> 2
        print("A.shape ", A.shape)
        print("shape ", A[:, 0, :, :].shape)
        x = x + x * torch.unsqueeze(A[:, 0, :, :], dim=1)
        x_b = x_b + x_b * torch.unsqueeze(A[:, 1, :, :], dim=1)

        self.A = A.data.detach()

        return x, x_b


class FusionModule(nn.Module):

    def __init__(self, inplane_x, inplane_b, norm_layer=Norm2d, dr2=2, dr4=4):
        super(FusionModule, self).__init__()

        self.input_channels = inplane_x + inplane_b # x + y

        self.channels_single = int(self.input_channels / 4) # (x + y) / 4

        self.conv1 = nn.Sequential( # (x + y) -> (x + y) / 4
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0, dilation=1),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential( # (x + y) / 4 -> (x + y) / 4
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, dilation=1),
            norm_layer(self.channels_single), nn.ReLU(inplace=True))


        self.ca = ChannelAttention(self.channels_single)
        self.sa = SpatialAttention()

        self.channel_reduction = nn.Sequential( # (x + y) / 4 -> 2
            nn.Conv2d(self.channels_single, 2, kernel_size=1))
    
    def forward(self, x, x_b): # [seg, edge]
        out = torch.cat([x, x_b], dim=1)

        out = self.conv1(out)
        out = self.conv2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        A = torch.sigmoid(self.channel_reduction(out)) # (x + y) / 4 -> 2
        x = x + x * torch.unsqueeze(A[:, 0, :, :], dim=1)
        x_b = x_b + x_b * torch.unsqueeze(A[:, 1, :, :], dim=1)

        return x, x_b


def main():
    b, c, h, w = 16, 384, 32, 32
    
    res = 0
    while c >= 48:
        x = torch.randn(b, c, h, w).cuda()
        x_b = torch.randn(b, c, h, w).cuda()

        model = FusionModule(c, c)
        summary(model, input_size=[(b,c,h,w), (b,c,h,w)])

        res = res + sum([param.nelement() for param in model.parameters()])
        c = int(c / 2)
        h = int(h / 2)
        w = int(w / 2)

    # x, x_b = model(x, x_b)
    # print("x.shape ", x.shape)
    # print("x_b.shape ", x_b.shape)
    # out = model(x, x_b)
    # print("out.shape ", out.shape)
    
    print("%fM params" % (res / 1e6))

if __name__ == '__main__':
    main()