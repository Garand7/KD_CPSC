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

np.random.seed(1337)
eporch = 500

batch_size = 512 * 4

XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)

YTrain_lstm_all = np.load('YTrain_Bilstm.npy', allow_pickle=True)

row_rand = np.arange(XTrain.shape[0])
np.random.shuffle(row_rand)

validation_size = (XTrain.shape[0]//10)*9
XTrain_all = XTrain[row_rand[0:validation_size]]
YTrain_all = YTrain[row_rand[0:validation_size]]
YTrain_lstm_all_f = YTrain_lstm_all[row_rand[0:validation_size]]

XTest = XTrain[row_rand[validation_size:]]
YTest = YTrain[row_rand[validation_size:]]
YTest_lstm_all_f = YTrain_lstm_all[row_rand[validation_size:]]

XTrain_abnormal = []
YTrain_abnormal = []
for i in range(len(YTrain_all)):
    if YTrain_all[i] == 1:
        XTrain_abnormal.append(XTrain_all[i])
        YTrain_abnormal.append(YTrain_lstm_all_f[i])

XTest_abnormal = []
YTest_abnormal = []
for i in range(len(YTest)):
    if YTest[i] == 1:
        XTest_abnormal.append(YTest[i])
        YTest_abnormal.append(YTest_lstm_all_f[i])

XTrain_abnormal = torch.from_numpy(np.array(XTrain_abnormal)).float()
YTrain_abnormal = torch.from_numpy(np.array(YTrain_abnormal)).long()

valid_num = len(XTrain_abnormal)//10

for valid_index in range(1):
    XValid = XTrain_abnormal[valid_num * valid_index:valid_num * (valid_index + 1)]
    YValid = YTrain_abnormal[valid_num * valid_index:valid_num * (valid_index + 1)]
    temp = list(range(len(XTrain_abnormal)))
    del temp[valid_num * valid_index:valid_num * (valid_index + 1)]
    XTrain_actual = XTrain_abnormal[temp]
    YTrain_actual = YTrain_abnormal[temp]

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

    cell = 200

    rnn = RNN(XTrain.shape[2], cell, 3).cuda()
    # rnn = nn.DataParallel(rnn)
    # rnn = rnn.cuda()
    print(rnn)


    optimizer = torch.optim.Adam(rnn.parameters())
    loss_func = nn.NLLLoss()
    accuracy = [0]
    accuracy_V = [0]
    accuracy_S = [0]

    max_accuracy = 0
    dt = datetime.datetime.now()
    model_dir = 'New_model_Bilstm{:0>2d}{:0>2d}{:0>2d}_cell{:0d}_{:0>2d}'.format(dt.month, dt.day, dt.hour, cell, valid_index)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    for epoch in range(21):
        print('epoch:%d' % epoch)
        for step, (b_x, b_y) in enumerate(loader):
            print('step:%d' % step)
            output = rnn(b_x.cuda())
            output = output.reshape(output.shape[0]*output.shape[1], output.shape[2])
            b_y = b_y.reshape(b_y.shape[0]*b_y.shape[1], 1)
            # loss = loss_func(output, b_y.cuda())
            loss = FocalLoss(gamma=2, alpha=[0.01, 0.5, 0.5])(output, b_y.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.item())

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
        result = temp == YValid.numpy()
        temp_sum = np.sum(result)
        temp_size = np.size(result)
        accuracy.append(temp_sum/temp_size)
        print('accuracy:%f' % accuracy[-1])

        V_right = 0
        V_num = 0
        S_right = 0
        S_num = 0
        for i in range(YValid.shape[0]):
            for j in range(YValid.shape[1]):
                if YValid[i][j] == 1:
                    V_num += 1
                    V_right += temp[i][j] == 1
                elif YValid[i][j] == 2:
                    S_num += 1
                    S_right += temp[i][j] == 2

        accuracy_V.append(V_right/V_num)
        accuracy_S.append(S_right / S_num)
        print('accuracy_V:%f' % (accuracy_V[-1]))
        print('accuracy_S:%f' % (accuracy_S[-1]))

        if max_accuracy < accuracy[-1]:
            torch.save(rnn, './' + model_dir + '/epoch' + str(epoch) + '_'
                       + str(accuracy[-1])[0:7] + '_' +
                       str(V_right/V_num)[1:7] + '_'
                       + str(S_right / S_num)[1:7] + '.pth')
            max_accuracy = accuracy[-1]
        if epoch % 20 == 0:
            torch.save(rnn, './' + model_dir + '/epoch' + str(epoch) + '_'
                       + str(accuracy[-1])[0:7] + '_' +
                       str(V_right / V_num)[1:7] + '_'
                       + str(S_right / S_num)[1:7] + '.pth')
    np.save('./' + model_dir + '/accuracy', accuracy)
    np.save('./' + model_dir + '/accuracy_S', accuracy_S)
    np.save('./' + model_dir + '/accuracy_V', accuracy_V)

