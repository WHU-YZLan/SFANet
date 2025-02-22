import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import numpy as np
from functools import reduce
from src.resnet import ResNet50, ResNet18
from torchvision import models as models
np.random.seed(2020)
# torch.manual_seed(3407)
from src.res2net_v1b_base import Res2Net_model

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) >> 1
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[: kernel_size, : kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = filt
    return torch.from_numpy(weight)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, groups=1, last_relu=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        modules = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
            nn.GroupNorm(mid_channels // 16, mid_channels, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
            nn.GroupNorm(out_channels // 16, out_channels, eps=1e-5)
        ]
        if last_relu:
            modules.append(nn.ReLU(inplace=True))

        self.double_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()

        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups),
            nn.GroupNorm(8, out_channels, eps=1e-5)
        ]

        if last_relu:
            modules.append(nn.ReLU(inplace=True))

        self.single_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, None, groups=groups, last_relu=last_relu)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ReverseDown(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels + (in_channels << 1), out_channels, None, groups=groups, last_relu=last_relu)

    def forward(self, x1, x2):
        x1 = self.down(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + (in_channels >> 1), out_channels, None, groups=groups, last_relu=last_relu)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpAlone(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, None, groups=groups, last_relu=last_relu)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class ReverseUp(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, None, groups=groups, last_relu=last_relu)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GCN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(GCN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 7), padding=(0, 3))
        self.conv12 = nn.Conv2d(n_classes, n_classes, kernel_size=(7, 1), padding=(3, 0))

        self.conv21 = nn.Conv2d(n_classes, n_classes, kernel_size=(1, 7), padding=(0, 3))
        self.conv22 = nn.Conv2d(in_channels, n_classes, kernel_size=(7, 1), padding=(3, 0))
    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)

        x2 = self.conv22(x)
        x2 = self.conv21(x2)

        out = x1 + x2

        return out
class ASE(nn.Module):
    def __init__(self, in_channels, n_classes, attention_channel):
        super(ASE, self).__init__()
        self.ac = attention_channel
        self.C = GCN(in_channels, n_classes)
    def forward(self, feature, mask):
        max_probs, EA_weight = torch.max(torch.softmax(mask, dim=1), dim=1)
        max_probs[torch.logical_and(max_probs > 0.4, EA_weight != 0)] = 0
        redined_feature = feature * max_probs[:, None, :, :]
        attention_map = torch.zeros_like(mask)
        attention_map[:, self.ac, :, :] = self.C(redined_feature)[:, self.ac, :, :]
        prediction_map = attention_map + mask
        return prediction_map

class SAF(nn.Module):
    def __init__(self, in_channels, t_attention_channel, spec_attention_channel, n_class=6):
        super(SAF, self).__init__()
        self.ac_T = t_attention_channel
        self.ac_S = spec_attention_channel
        self.out = OutConv(in_channels, n_class)
        # self.decode = DoubleConv(in_channels, 64)
        # self.CA = ChannelAttentionModule(in_channels)
    def forward(self, main_branch_feature, t_feature, spec_feature):
        mask = self.out(main_branch_feature)
        max_probs, EA_weight = torch.max(torch.softmax(mask, dim=1), dim=1)
        t_max_probs = max_probs.clone()
        spec_max_probs = max_probs.clone()
        t_max_probs[torch.logical_or(max_probs < 0.4, EA_weight != self.ac_T)] = 0
        t_redined_feature = t_feature * t_max_probs[:, None, :, :]
        spec_max_probs[torch.logical_or(max_probs < 0.4, EA_weight != self.ac_S)] = 0
        spec_refined_feature = spec_feature * spec_max_probs[:, None, :, :]
        return t_redined_feature, spec_refined_feature, mask



class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class Up_skip(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bilinear=True):
        super(Up_skip, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, groups=groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, groups=groups)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x






class SFANet(nn.Module):
    def __init__(self, n_classes):
        super(SFANet, self).__init__()

        resnet_raw_model1 = models.resnet50(pretrained=True)
        resnet_raw_model2 = models.resnet50(pretrained=True)
        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = resnet_raw_model1.conv1
        self.encoder_thermal_conv1.weight.data = self.encoder_thermal_conv1.weight.data[:, :1, :, :]
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4


        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        # self.encoder_rgb_conv1.weight.data= torch.cat([self.encoder_rgb_conv1.weight.data, self.encoder_rgb_conv1.weight.data], dim=1)
        # self.encoder_rgb_conv1.weight.data = self.encoder_rgb_conv1.weight.data[:, :5, :, :]
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        del resnet_raw_model1, resnet_raw_model2

        self.att_T = 0
        self.att_S = 1

        self.compression = nn.Sequential(
            nn.Conv2d(4096, 2048, 1, 1, 0, bias=False, groups=1),
            nn.GroupNorm(2048 // 16, 2048, eps=1e-5),
            nn.ReLU(inplace=True),
        )


        self.outc = OutConv(64, n_classes)

        # 1103 add
        self.skip_up_d1 = Up_skip(3072, 256*4)
        self.skip_up_d2 = Up_skip(1536, 128*4)
        self.skip_up_d3 = Up_skip(768, 64*4)
        self.skip_up_d4 = Up_skip(320, 64)
        self.upsample_t = Up_skip(64+64, 64)
        self.upsample_spec = Up_skip(64+64, 64)
        self.upsample_f = Up_skip(64+64, 64)
        # self.skip_up_d4 = Up_skip(128, 64)
        # self.d_out = OutConv(64, n_classes)


        self.fuse4 = nn.Sequential(
            nn.Conv2d(512*4, 256*4, 1, 1, 0, bias=False, groups=1),
            nn.GroupNorm(256*4 // 16, 256*4, eps=1e-5),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(256*4, 128*4, 1, 1, 0, bias=False, groups=1),
            nn.GroupNorm(128*4 // 16, 128*4, eps=1e-5),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128*4, 64*4, 1, 1, 0, bias=False, groups=1),
            nn.GroupNorm(64*4 // 16, 64*4, eps=1e-5),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64*2, 64, 1, 1, 0, bias=False, groups=1),
            nn.GroupNorm(64 // 16, 64, eps=1e-5),
            nn.ReLU(inplace=True),
        )

        self.trans = nn.Conv2d(128, 64, 1, 1)
        #1111 add
        self.EA_T_1 = SAF(64, self.att_T, self.att_S, n_classes)
        self.EA_T_2 = SAF(256, self.att_T, self.att_S, n_classes)
        self.EA_T_3 = SAF(512, self.att_T, self.att_S, n_classes)
        self.EA_T_4 = SAF(1024, self.att_T, self.att_S, n_classes)
        self.EA_T_5 = SAF(2048, self.att_T, self.att_S, n_classes)

        self.EA_T_Out = ASE(64, n_classes, self.att_T)
        self.EA_S_Out = ASE(64, n_classes, self.att_S)
        self.convt = DoubleConv(1, 64, 32)
        self.convs = DoubleConv(3, 64, 32)
        self.CAt = ChannelAttentionModule(64)
        self.CAs = ChannelAttentionModule(64)
        # self.trans2 = nn.Conv2d(5, 3, 1, 1)

    def forward(self, rgb, t):
        # spec[:, :3, :, :] = rgb
        # spec = self.trans2(spec)
        # t = torch.cat([t, t, t], dim=1)
        tc = self.convt(t)
        tc = self.CAt(tc) * tc
        sc = self.convs(rgb)
        sc = self.CAs(sc) * sc
        t1 = self.encoder_thermal_conv1(t)
        t1 = self.encoder_thermal_bn1(t1)
        t1 = self.encoder_thermal_relu(t1)
        t2 = self.encoder_thermal_maxpool(t1)
        t2 = self.encoder_thermal_layer1(t2)
        t3 = self.encoder_thermal_layer2(t2)
        t4 = self.encoder_thermal_layer3(t3)
        t5 = self.encoder_thermal_layer4(t4)
        spec1 = self.encoder_rgb_conv1(rgb)
        spec1 = self.encoder_rgb_bn1(spec1)
        spec1 = self.encoder_rgb_relu(spec1)
        spec2 = self.encoder_rgb_maxpool(spec1)
        spec2 = self.encoder_rgb_layer1(spec2)
        spec3 = self.encoder_rgb_layer2(spec2)
        spec4 = self.encoder_rgb_layer3(spec3)
        spec5 = self.encoder_rgb_layer4(spec4)

        f1 = self.fuse1(torch.cat([t1, spec1], dim=1))
        EA_S, EA_T, out_d1 = self.EA_T_1(f1, t1, spec1)
        f1 = f1 + EA_S + EA_T
        f2 = self.fuse2(torch.cat([t2, spec2], dim=1))
        EA_S, EA_T, out_d2 = self.EA_T_2(f2, t2, spec2)
        f2 = f2 + EA_S + EA_T
        f3 = self.fuse3(torch.cat([t3, spec3], dim=1))
        EA_S, EA_T, out_d3 = self.EA_T_3(f3, t3, spec3)
        f3 = f3 + EA_S + EA_T
        f4 = self.fuse4(torch.cat([t4, spec4], dim=1))
        EA_S, EA_T, out_d4 = self.EA_T_4(f4, t4, spec4)
        f4 = f4 + EA_S + EA_T

        x = self.compression(torch.cat([t5, spec5], dim=1))
        EA_S, EA_T, out_d5 = self.EA_T_5(x, t5, spec5)
        x = x + EA_S + EA_T
        x = self.skip_up_d1(x, f4)
        x = self.skip_up_d2(x, f3)
        x = self.skip_up_d3(x, f2)
        x = self.skip_up_d4(x, f1)

        x = self.upsample_f(x, self.trans(torch.cat([tc, sc], dim=1)))

        mask_t = self.upsample_t(t1, tc)

        mask_spec = self.upsample_spec(spec1, sc)

        out_x = self.outc(x)
        logits = out_x + self.EA_T_Out(mask_t, out_x) + self.EA_S_Out(mask_spec, out_x)
        # return {"out1": logits}
        return {"out1": logits, "out_x": out_x, "down1": out_d1, "down2": out_d2,
                "down3": out_d3, "down4": out_d4, "down5": out_d5}

# net = MEPDNet(n_channels1=3, n_channels2=3, n_channels3=5, n_classes=2)
# print(net)

if __name__ == '__main__':
    # main()
    model = MEPDNet(n_channels1=3, n_channels2=3, n_channels3=5, n_classes=6)
    toyal_params = sum(p.numel() for p in model.parameters())
    print(f"Total is : {toyal_params}")