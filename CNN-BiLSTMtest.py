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

XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain_Bilstm.npy', allow_pickle=True)
YTrain_CNN = np.load('YTrain.npy', allow_pickle=True)

XTrain_lstm = torch.from_numpy(XTrain).float()
XTrain_CNN = torch.from_numpy(np.expand_dims(XTrain, 1)).float()
YTrain = torch.from_numpy(YTrain).long()
YTrain_CNN = torch.from_numpy(YTrain_CNN).long()

row_rand = np.arange(XTrain_CNN.shape[0])
np.random.shuffle(row_rand)

validation_size = (XTrain_CNN.shape[0] // 10) * 9

XTest_CNN = XTrain_CNN[row_rand[validation_size:]]
YTest_CNN = YTrain_CNN[row_rand[validation_size:]]
XTest_lstm = XTrain_lstm[row_rand[validation_size:]]
YTest = YTrain[row_rand[validation_size:]]



batch_size = 256 * 2
kernel_size = 4
stride = 2
padding = 0
ndf = 18
alpha = 0.01


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


Model_CNN = torch.load('/home/yinyibo/PycharmProjects/pytorch/ECG/' +
                    'code/11.0/ICCSN/paper/CNN_validation_best/9_epoch120_0.94708_0.9480_0.9416.pth')
rnn = torch.load('/home/yinyibo/PycharmProjects/pytorch/ECG/code/'+
                 '11.0/ICCSN/paper/A_New_BiLSTM_validation/9_epoch100_0.98795_0.9719_0.9470.pth')


CNN_abnormal_num = 0

result_CNN = []
Model_CNN.eval()#This is very important. The function of this sentence is to let the BN layer no longer update the parameters during the later model operation.
#Since the batch size of this test is 1 each time, the performance of the model will be much worse without this sentence.

for i in range(len(XTest_CNN)):
    # print('index %d' % i)
    x = XTest_CNN[i].unsqueeze(0).cuda()
    x2 = XTest_CNN[i].cuda()
    pred_cnn = torch.max(Model_CNN(x), 1)[1]
    if pred_cnn == 0:
        result_CNN.append(0)
        if i == 0:
            pred_test = torch.zeros((1, XTrain.shape[1])).cuda()
            pred_test_back = torch.max(rnn(x2), 2)[1]
        else:
            pred_test = torch.cat((pred_test, torch.zeros((1, XTrain.shape[1])).cuda()), dim=0)
            temp = torch.max(rnn(x2), 2)[1]
            pred_test_back = torch.cat((pred_test_back, temp), dim=0)
    else:
        result_CNN.append(1)
        CNN_abnormal_num += 1
        if i == 0:
            pred_test = torch.max(rnn(x2), 2)[1]
            pred_test_back = torch.max(rnn(x2), 2)[1]
        else:
            temp = torch.max(rnn(x2), 2)[1]
            pred_test = torch.cat((pred_test, temp), dim=0)
            pred_test_back = torch.cat((pred_test_back, temp), dim=0)


result_CNN = np.array(result_CNN)

CNN_result = YTest_CNN.numpy() == result_CNN
temp = np.sum(CNN_result)/np.size(CNN_result)
print('cnn acc:%Ff' % temp)

CNN_0_num = 0
CNN_1_num = 0
True_CNN_0_num = 0
True_CNN_1_num = 0
Right_CNN_0_num = 0
Right_CNN_1_num = 0
for i in range(len(result_CNN)):
    if result_CNN[i] == 0:
        CNN_0_num += 1
        if YTest_CNN[i] == 0:
            True_CNN_0_num += 1
            Right_CNN_0_num += 1
            continue
        else:
            True_CNN_1_num += 1
            continue
    else:
        CNN_1_num += 1
        if YTest_CNN[i] == 1:
            True_CNN_1_num += 1
            Right_CNN_1_num += 1
            continue
        else:
            True_CNN_0_num += 1
            continue

print('CNN_0_num:%d' % (CNN_0_num))
print('CNN_1_num:%d' % (CNN_1_num))
print('True_CNN_0_num:%d' % (True_CNN_0_num))
print('True_CNN_1_num:%d' % (True_CNN_1_num))
print('Right_CNN_0_num:%d' % (Right_CNN_0_num))
print('Right_CNN_1_num:%d' % (Right_CNN_1_num))


print('total_CNN_num:%d' % len(XTest_CNN))
print('CNN_abnormal_num:%d' % CNN_abnormal_num)

# pred_test = pred_test_back
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



