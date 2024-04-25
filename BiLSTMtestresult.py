import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

np.random.seed(1337)

batch_size = 512 * 4
#zheng chang cao zuo
# XTrain_abnormal = np.load('XTrain_abnormal_lstm.npy',
#              allow_pickle=True)
# YTrain_abnormal = np.load('YTrain_abnormal_lstm.npy',
#              allow_pickle=True)
#
# XTrain = torch.from_numpy(XTrain_abnormal).float()
# YTrain = torch.from_numpy(YTrain_abnormal).long()

XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain_Bilstm.npy', allow_pickle=True)
XTrain = torch.from_numpy(XTrain).float()
YTrain = torch.from_numpy(YTrain).long()

row_rand = np.arange(XTrain.shape[0])
np.random.shuffle(row_rand)

validation_size = (XTrain.shape[0]//10)*9
XTrain_all = XTrain[row_rand[0:validation_size]]
YTrain_all = YTrain[row_rand[0:validation_size]]

XTest = XTrain[row_rand[validation_size:]]
YTest = YTrain[row_rand[validation_size:]]

torch_testset = Data.TensorDataset(XTest, YTest)
loader2 = Data.DataLoader(
    dataset=torch_testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

class RNN(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=nIn,
            hidden_size=nHidden,
            bidirectional=True,
            batch_first=True
        )
        self.out = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        out, (h, c) = self.rnn(x)
        out = self.out(out)
        return out


rnn = torch.load('/home/yinyibo/PycharmProjects/pytorch/ECG/code/'+
                 '11.0/ICCSN/A_New_BiLSTM_validation/9_epoch100_0.98795_0.9719_0.9470.pth')

# print(rnn)
# # summary(VGGSPP, (1, XTrain.shape[2], XTrain.shape[3]))
# input_t = torch.randn(1, XTrain.shape[1], XTrain.shape[2]).cuda()
# flops, params = profile(rnn, inputs=(input_t, ))
# print(flops, params)


for step2, (batch_x2, batch_y2) in enumerate(loader2):
    print('valid:step %d' % step2)
    x2 = batch_x2.cuda()
    if step2 == 0:
        pred_test = torch.max(rnn(x2), 2)
        pred_test = pred_test[1]
    else:
        pred_test = torch.cat((pred_test, torch.max(rnn(x2), 2)[1]), dim=0)

pred_test = pred_test.cpu().numpy()
temp = np.expand_dims(pred_test, 2)
result = temp == YTest.numpy()
temp_sum = np.sum(result)
temp_size = np.size(result)
accuracy = (temp_sum/temp_size)
print('accuracy:%f' % accuracy)

V_num = 0
S_num = 0
V_true_num = 0
S_true_num = 0
V_right_num = 0
S_right_num = 0
for i in range(YTest.shape[0]):
    for j in range(YTest.shape[1]):
        if YTest[i][j] == 1:
            V_true_num += 1
            V_right_num += temp[i][j] == 1
        elif YTest[i][j] == 2:
            S_true_num += 1
            S_right_num += temp[i][j] == 2
        if temp[i][j] == 1:
            V_num += 1
        elif temp[i][j] == 2:
            S_num += 1
print('V_num:%d' % (V_num))
print('S_num:%d' % (S_num))
print('V_true_num:%d' % (V_true_num))
print('S_true_num:%d' % (S_true_num))
print('V_right_num:%d' % (V_right_num))
print('S_right_num:%d' % (S_right_num))
# V_right = 0
# V_num = 0
# S_right = 0
# S_num = 0
# n_num = 0
# n_right = 0
# for i in range(YTest.shape[0]):
#     for j in range(YTest.shape[1]):
#         if YTest[i][j] == 1:
#             V_num += 1
#             V_right += temp[i][j] == 1
#         elif YTest[i][j] == 2:
#             S_num += 1
#             S_right += temp[i][j] == 2
#         elif YTest[i][j] == 0:
#             n_num += 1
#             n_right += temp[i][j] == 0
#
# accuracy_V = (V_right/V_num)
# accuracy_S = (S_right / S_num)
# accuracy_N = n_right/n_num
# print('accuracy_V:%f' % (accuracy_V))
# print('accuracy_S:%f' % (accuracy_S))
# print('accuracy_N:%f' % (accuracy_N))
