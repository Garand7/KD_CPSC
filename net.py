import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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
# from focalloss import FocalLoss
from fcalloss import FocalLoss
from torchsummary import summary
from thop import profile
from matplotlib import pyplot as plt
from loguru import logger
start = time.time()
np.random.seed(1337)

ndf = 18
alpha = 0.01
XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)
elapsed = time.time() - start
# print("data load time used:", elapsed)
logger.info(f"数据加载时间：{elapsed}")
#%%
XTrain = torch.from_numpy(np.expand_dims(XTrain, 1)).float()
YTrain = torch.from_numpy(np.expand_dims(YTrain, 1)).long()

row_rand = np.arange(XTrain.shape[0])
np.random.shuffle(row_rand)

validation_size = (XTrain.shape[0]//10)*9
XTrain_all = XTrain[row_rand[0:validation_size]]
YTrain_all = YTrain[row_rand[0:validation_size]]

XTest = XTrain[row_rand[validation_size:]]
YTest = YTrain[row_rand[validation_size:]]

valid_num = len(XTrain_all)//10
#%%
for valid_index in range(1):
    XValid = XTrain_all[valid_num * valid_index:valid_num * (valid_index + 1)]
    YValid = YTrain_all[valid_num * valid_index:valid_num * (valid_index + 1)]
    temp = list(range(len(XTrain_all)))
    del temp[valid_num * valid_index:valid_num * (valid_index + 1)]
    XTrain_actual = XTrain_all[temp]
    YTrain_actual = YTrain_all[temp]

    elapsed = time.time() - start
    # print("data transform&choose and time used:", elapsed)
    logger.info(f"数据转换及选择用时：{elapsed}")

    start = time.time()

    batch_size = 256 * 2

    torch_dataset = Data.TensorDataset(XTrain_actual, YTrain_actual)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    torch_testset = Data.TensorDataset(XValid, YValid)
    loader2 = Data.DataLoader(
        dataset=torch_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    elapsed = time.time() - start
    # print("data to dataloader time used:", elapsed)
    logger.info(f"数据装载到 DataLoader 用时：{elapsed}")
#%%


# from TeacherNet import UPPS_NET
from unet import UNetCustom

# Model_CNN = UPPS_NET(1, ndf=ndf).cuda()
Model_CNN = UNetCustom(1,ndf=ndf).cuda()

print(Model_CNN)

summary(Model_CNN, (1, XTrain.shape[2], XTrain.shape[3])) #65 33
input_t = torch.randn(1, 1, XTrain.shape[2], XTrain.shape[3]).cuda()
flops, params = profile(Model_CNN, inputs=(input_t, ))
print(flops, params)