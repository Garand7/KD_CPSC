import math

import numpy as np
from thop import profile
from torchsummary import summary



"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ndf = 18

class UPPS_NET(nn.Module):
    def __init__(self, input_nc, ndf):
        super(UPPS_NET, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 2, bias=False)
        self.BN1 = nn.BatchNorm2d(ndf)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 2, groups=1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf, 4, 2, 2, groups=1, bias=False)
        self.fc1 = nn.Linear(ndf, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.BN1(x))

        x = self.conv2(x)
        x = F.leaky_relu(self.BN2(x))

        x = self.conv3(x)
        num_sample = x.size()[0]
        h_wid = int(math.ceil(int(x.size(2))))
        w_wid = int(math.ceil(int(x.size(3))))
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(0, 0))
        x = maxpool(x)
        x = x.view(num_sample, -1)
        fc1 = self.fc1(x)

        return fc1

Model_CNN = UPPS_NET(1, ndf=ndf).cuda()

print(Model_CNN)
ndf = 18
model = UPPS_NET(input_nc=1, ndf=ndf).cuda()

summary(model, (1, 32, 32))  # 假设输入大小为 32x32
