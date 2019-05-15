import torch
import torch.nn as nn
import torch.nn.functional as F


class Branch(nn.Module):
    def __init__(self, channels, dilated, kernel_size, norm='bn'):
        super(Branch, self).__init__()
        self.conv_n_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                  stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2), dilation=1, groups=1, bias=False)
        self.conv_1_n = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size[::-1],
                                  stride=1, padding=(kernel_size[1]//2, kernel_size[0]//2), dilation=1, groups=1, bias=False)

        self.conv_n_1_dilated = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                          stride=1, padding=(kernel_size[0]//2*dilated, kernel_size[1]//2*dilated), dilation=dilated, groups=1, bias=False)
        self.conv_1_n_dilated = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size[::-1],
                                          stride=1, padding=(kernel_size[1]//2*dilated, kernel_size[0]//2*dilated), dilation=dilated, groups=1, bias=False)

        if norm == 'gn':
            self.norm1 = nn.GroupNorm(num_groups=16, num_channels=channels)
            self.norm2 = nn.GroupNorm(num_groups=16, num_channels=channels)
        else:
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = F.relu(self.conv_n_1(x))
        x = self.conv_1_n(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = F.relu(self.conv_n_1_dilated(x))
        x = self.conv_1_n_dilated(x)
        x = self.norm2(x)
        x = F.relu(x)

        return x


class SS_nbt(nn.Module):
    def __init__(self, channels, dilated, norm='bn'):
        super(SS_nbt, self).__init__()
        self.channels = channels
        self.branch1 = Branch(channels=channels//2, dilated=dilated, kernel_size=(3, 1), norm=norm)
        self.branch2 = Branch(channels=channels//2, dilated=dilated, kernel_size=(1, 3), norm=norm)

    @staticmethod
    def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, x):
        # channels split
        x1 = x[:, :self.channels//2, :, :]
        x2 = x[:, self.channels//2:, :, :]

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        x = x + torch.cat([x1, x2], dim=1)
        x = F.relu(x)
        x = self.channel_shuffle(x, groups=2)

        return x


class DownSample(nn.Module):
    def __init__(self, inc, outc):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc//2, kernel_size=3, padding=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=inc, out_channels=outc//2, kernel_size=3, padding=2, stride=2)
        # TODO kernel_size isn't mentioned in paper
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool(x1)

        x2 = self.conv2(x)
        x2 = self.pool(x2)

        return torch.cat([x1, x2], dim=1)


class Encoder(nn.Module):
    def __init__(self, norm='bn'):
        super(Encoder, self).__init__()
        self.downsample1 = DownSample(3, 32)
        self.downsample2 = DownSample(32, 64)
        self.downsample3 = DownSample(64, 128)

        stage1 = []
        for i in range(3):
            stage1.append(SS_nbt(32, dilated=1, norm=norm))
        self.stage1 = nn.Sequential(*stage1)

        stage2 = []
        for i in range(2):
            stage2.append(SS_nbt(64, dilated=1, norm=norm))
        self.stage2 = nn.Sequential(*stage2)

        stage3 = []
        for dilated in [1, 2, 5, 9, 2, 5, 9, 17]:
            stage3.append(SS_nbt(128, dilated=dilated, norm=norm))
        self.stage3 = nn.Sequential(*stage3)

    def forward(self, x):
        x = self.downsample1(x)
        x = self.stage1(x)
        x = self.downsample2(x)
        x = self.stage2(x)
        x = self.downsample3(x)
        x = self.stage3(x)

        return x


class Conv2dNormReLu(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding, norm='bn'):
        super(Conv2dNormReLu, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size, stride=stride, padding=padding)
        if norm == 'gn':
            self.norm = nn.GroupNorm(16, outc)
        else:
            self.norm = nn.BatchNorm2d(outc)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, with_norm_relu=True):
        super(Decoder, self).__init__()
        conv = Conv2dNormReLu if with_norm_relu else nn.Conv2d

        # pooling
        self.conv1 = nn.Conv2d(128, num_classes, 1, 1)

        # 128*64
        self.conv2 = nn.Conv2d(128, num_classes, 1, 1)

        # 64*32
        self.conv_7_7 = conv(128, 128, 7, 2, 3)
        self.conv3 = nn.Conv2d(128, num_classes, 1, 1)

        # 32*16
        self.conv_5_5 = conv(128, 128, 5, 2, 2)
        self.conv4 = nn.Conv2d(128, num_classes, 1, 1)

        # 16*8
        self.conv_3_3 = conv(128, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, num_classes, 1, 1)

    def forward(self, x):
        branch1 = F.adaptive_avg_pool2d(x, (1, 1))
        branch1 = self.conv1(branch1)
        branch1 = F.interpolate(branch1, (64, 128), mode='bilinear', align_corners=True)

        branch2 = self.conv2(x)

        branch3to4 = self.conv_7_7(x)
        branch3 = self.conv3(branch3to4)

        branch4to5 = self.conv_5_5(branch3to4)
        branch4 = self.conv4(branch4to5)

        branch5 = self.conv_3_3(branch4to5)
        branch5 = self.conv5(branch5)
        branch5 = F.interpolate(branch5, scale_factor=2, mode='bilinear', align_corners=True)

        branch4 = branch4 + branch5
        branch4 = F.interpolate(branch4, scale_factor=2, mode='bilinear', align_corners=True)

        branch3 = branch3 + branch4
        branch3 = F.interpolate(branch3, scale_factor=2, mode='bilinear', align_corners=True)

        branch2 = branch2 * branch3

        logits = branch2 + branch1
        logits = F.interpolate(logits, scale_factor=8, mode='bilinear', align_corners=True)

        return logits


class LEDNet(nn.Module):
    def __init__(self, num_classes, norm='bn', with_norm_relu=True):
        super(LEDNet, self).__init__()
        self.encoder = Encoder(norm)
        self.decoder = Decoder(num_classes, with_norm_relu)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    from utils import params_size
    x = torch.rand(1, 3, 512, 1024).cuda()
    net = LEDNet(19, with_norm_relu=False).cuda()
    params_size(net)
    import time
    t1 = time.time()
    x = net(x)
    t2 = time.time()
    print(t2-t1)