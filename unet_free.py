import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt  # 用于小波变换
import numpy as np
import torch.utils.data as Data
from torchsummary import summary


def dwt_transform(x, level=1):
    # 使用 wavedec2 进行小波变换
    coeffs = pywt.wavedec2(x.detach().cpu().numpy(), 'haar', level=level)
    # 将小波系数转换为数组
    coeffs_array, coeffs_slices = pywt.coeffs_to_array(coeffs, axes=(0, 1))
    # 将数组转换为张量
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(x.device)
    return coeffs_tensor, coeffs_slices

def idwt_transform(coeffs_tensor, coeffs_slices):
    # 将张量转换回 numpy 数组
    coeffs_array = coeffs_tensor.detach().cpu().numpy()
    # 将数组转换回小波系数
    coeffs = pywt.array_to_coeffs(coeffs_array, coeffs_slices, output_format='wavedec2')
    recon = pywt.waverec2(coeffs, 'haar')
    return torch.tensor(recon, dtype=torch.float32).to(coeffs_tensor.device)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UNetCustom(nn.Module):
    def review_module(self, x, target_channels):
        review_conv = nn.Conv2d(x.size(1), target_channels, kernel_size=1).to(x.device)
        x = F.relu(review_conv(x))
        return x

    def __init__(self, input_nc, ndf):
        super(UNetCustom, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.se1 = SEBlock(ndf)
        self.enc2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.se2 = SEBlock(ndf * 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(ndf * 2, 2)  # 输出2个类别
        self.stage_outputs = {}  # 用于存储中间层输出

    def forward(self, x):
        x1 = self.enc1(x)
        self.stage_outputs['enc1'] = x1
        x1 = self.se1(x1)

        x2 = self.enc2(x1)
        self.stage_outputs['enc2'] = x2
        x2 = self.se2(x2)

        # 频率变换
        x2_freq, x2_slices = dwt_transform(x2)
        # 特征回顾和融合
        if 'enc1' in self.stage_outputs:
            reviewed_x1 = self.review_module(self.stage_outputs['enc1'], x2.size(1))
            reviewed_x1 = F.interpolate(reviewed_x1, size=(x2.size(2), x2.size(3)))
            reviewed_x1_freq, _ = dwt_transform(reviewed_x1)
            x2_freq += reviewed_x1_freq  # 融合回顾的特征和当前特征的频率表示

        # 逆频率变换
        x2_recon = idwt_transform(x2_freq, x2_slices)

        x = self.global_avg_pool(x2_recon)  # 使用重建后特征
        x = x.view(x.size(0), -1)

        out = self.final(x)
        return x2, out


# 模型初始化和测试
ndf = 18
model = UNetCustom(input_nc=1, ndf=ndf).cuda()
summary(model, (1, 32, 32))  # 输入大小为 1x32x32