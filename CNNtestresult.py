#%%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import torch.utils.data as Data
import time
import os
import datetime
from focalloss import FocalLoss
from torchsummary import summary
from thop import profile
from matplotlib import pyplot as plt

start = time.time()
np.random.seed(1337)

kernel_size = 4
stride = 2
padding = 0
ndf = 18
alpha = 0.01

XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)
elapsed = time.time() - start
print("data load time used:", elapsed)

XTrain = torch.from_numpy(np.expand_dims(XTrain, 1)).float()
YTrain = torch.from_numpy(np.expand_dims(YTrain, 1)).long()

row_rand = np.arange(XTrain.shape[0])
np.random.shuffle(row_rand)

validation_size = (XTrain.shape[0] // 10) * 9
XTrain_all = XTrain[row_rand[0:validation_size]]
YTrain_all = YTrain[row_rand[0:validation_size]]

XTest = XTrain[row_rand[validation_size:]]
YTest = YTrain[row_rand[validation_size:]]

batch_size = 1

torch_testset = Data.TensorDataset(XTest, YTest)
loader2 = Data.DataLoader(
    dataset=torch_testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)


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
# 初始化模型
ndf = 18
Model_CNN = UPPS_NET(1, ndf=ndf).cuda()
model_path = './train_parallel_18_0.010000_00/epoch200_0.96132_0.9623_0.9557.pth'
state_dict = torch.load(model_path)
# 删除不是模型参数的键
state_dict.pop('total_ops', None)  # 删除 'total_ops'，如果存在
state_dict.pop('total_params', None)  # 删除 'total_params'，如果存在
Model_CNN.load_state_dict(state_dict)
#%%

Model_CNN.eval()
#%%
print(Model_CNN)

summary(Model_CNN, (1, XTrain.shape[2], XTrain.shape[3]))  # 65 33
input_t = torch.randn(1, 1, XTrain.shape[2], XTrain.shape[3]).cuda()
flops, params = profile(Model_CNN, inputs=(input_t,))
print(flops, params)

for step2, (batch_x2, batch_y2) in enumerate(loader2):
    print('valid:step %d' % step2)
    x2 = batch_x2.cuda()
    if step2 == 0:
        pred_test = torch.max(Model_CNN(x2), 1)
        pred_test = pred_test[1]
    else:
        pred_test = torch.cat((pred_test, torch.max(Model_CNN(x2), 1)[1]), dim=0)

pred_test = pred_test.cpu().numpy()
temp = np.expand_dims(pred_test, 1) == YTest.numpy()
accuracy = np.sum(temp) / np.size(temp)

print('accuracy_all:%f' % accuracy)

abnormal_right = 0
normal_right = 0
for i in range(len(pred_test)):
    if YTest[i] == 1:
        abnormal_right += pred_test[i] == 1
    else:
        normal_right += pred_test[i] == 0

accuracy_normal = (normal_right / (len(YTest) - sum(YTest)))
accuracy_abnormal = (abnormal_right / (sum(YTest)))
print('accuracy_normal:%f' % (accuracy_normal))
print('accuracy_abnormal:%f' % (accuracy_abnormal))

