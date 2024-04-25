import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_nc, ndf):
        super(UNetCustom, self).__init__()
        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),  # 减少到理想大小
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.se1 = SEBlock(ndf)  # 注意力模块
        self.enc2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),  # 维持尺寸
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.se2 = SEBlock(ndf * 2)  # 注意力模块

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 最终输出层
        self.final = nn.Linear(ndf * 2, 2)  # 输出2个类别

    def forward(self, x):
        x = self.enc1(x)  # 经过第一层后尺寸减半
        x = self.se1(x)  # 经过第一个SE块
        x = self.enc2(x)  # 第二层保持尺寸
        x = self.se2(x)  # 经过第二个SE块
        x = self.global_avg_pool(x)  # 全局平均池化到1x1
        x = x.view(x.size(0), -1)  # 展平
        out = self.final(x)  # 经过全连接层得到最终输出
        return out

# 模型初始化和测试
ndf = 18
model = UNetCustom(input_nc=1, ndf=ndf).cuda()
print(model)

from torchsummary import summary
summary(model, (1, 32, 32))  # 输入大小为 1x32x32