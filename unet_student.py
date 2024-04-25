import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetCustom(nn.Module):
    def __init__(self, input_nc, ndf):
        super(UNetCustom, self).__init__()
        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),  # 减少到理想大小
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),  # 维持尺寸
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 最终输出层
        self.final = nn.Linear(ndf * 2, 2)  # 输出2个类别

    def forward(self, x):
        x = self.enc1(x)  # 经过第一层后尺寸减半
        x = self.enc2(x)  # 第二层保持尺寸
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