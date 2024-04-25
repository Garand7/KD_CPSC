import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetCustom(nn.Module):
    def __init__(self, input_nc, ndf):
        super(UNetCustom, self).__init__()
        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 解码器部分
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 最终输出层，不变
        self.final = nn.Linear(ndf, 2)

    def forward(self, x):
        # 编码器路径
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 解码器路径
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)

        # 跳跃连接和输出
        out = d2 + e1  # 跳跃连接，确保维度匹配
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.final(out)

        return out


# 模型初始化
ndf = 18
model = UNetCustom(input_nc=1, ndf=ndf).cuda()
print(model)

from torchsummary import summary

summary(model, (1, 32, 32))  # 假设输入大小为 32x32