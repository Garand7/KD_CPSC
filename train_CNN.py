import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pandas as pd

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
from unet_student import UNetCustom
start = time.time()
np.random.seed(1337)

ndf = 18
alpha = 0.01
XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)
elapsed = time.time() - start
print("data load time used:", elapsed)

def save_metrics_to_excel(accuracy, accuracy_normal, accuracy_abnormal, model_dir):
    # 创建一个 DataFrame
    df = pd.DataFrame({
        'Epoch': range(len(accuracy)),
        'Overall Accuracy': accuracy,
        'Normal Accuracy': accuracy_normal,
        'Abnormal Accuracy': accuracy_abnormal
    })
    # 指定保存目录
    excel_path = os.path.join(model_dir, 'training_metrics.xlsx')
    # 保存到 Excel 文件
    df.to_excel(excel_path, index=False)
    print(f"Metrics saved to {excel_path}")
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

for valid_index in range(1):
    XValid = XTrain_all[valid_num * valid_index:valid_num * (valid_index + 1)]
    YValid = YTrain_all[valid_num * valid_index:valid_num * (valid_index + 1)]
    temp = list(range(len(XTrain_all)))
    del temp[valid_num * valid_index:valid_num * (valid_index + 1)]
    XTrain_actual = XTrain_all[temp]
    YTrain_actual = YTrain_all[temp]

    elapsed = time.time() - start
    print("data transform&choose and time used:", elapsed)
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
    print("data to dataloader time used:", elapsed)





    Model_CNN = UNetCustom(1, ndf=ndf).cuda()

    print(Model_CNN)

    summary(Model_CNN, (1, XTrain.shape[2], XTrain.shape[3])) #65 33
    input_t = torch.randn(1, 1, XTrain.shape[2], XTrain.shape[3]).cuda()
    flops, params = profile(Model_CNN, inputs=(input_t, ))
    print(flops, params)

    optimizer = torch.optim.AdamW(Model_CNN.parameters())
#     loss_func = nn.BCEWithLogitsLoss()

    accuracy = [0]
    accuracy_normal = [0]
    accuracy_abnormal = [0]
    max_accuracy = 0
    loss_all = [0]
    dt = datetime.datetime.now()

    model_dir = 'train_unet_student_{:0>2d}_{:0>4f}_{:0>2d}'.format(ndf, alpha, valid_index)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)



    for epoch in range(201):
        start = time.time()
        print('epoch:%d' % epoch)
        for step, (batch_x, batch_y) in enumerate(loader):
            x = batch_x.cuda()
            y = batch_y.cuda().squeeze()
            print('step:%d' % step)
            out = Model_CNN(x)
            # loss = loss_func(out, y)
            loss = FocalLoss(gamma=2, alpha=[alpha, 1])(out, y)
            loss_all.append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for step2, (batch_x2, batch_y2) in enumerate(loader2):
            print('valid:step %d' % step2)
            x2 = batch_x2.cuda()
            if step2 == 0:
                pred_test = torch.max(Model_CNN(x2), 1)
                pred_test = pred_test[1]
            else:
                pred_test = torch.cat((pred_test, torch.max(Model_CNN(x2), 1)[1]), dim=0)

        pred_test = pred_test.cpu().numpy()
        temp = np.expand_dims(pred_test, 1) == YValid.numpy()
        temp = np.sum(temp)/np.size(temp)
        accuracy.append(temp)
        print('accuracy_all:%f' % accuracy[-1])

        abnormal_right = 0
        normal_right = 0
        for i in range(len(pred_test)):
            if YValid[i] == 1:
                abnormal_right += pred_test[i] == 1
            else:
                normal_right += pred_test[i] == 0

        accuracy_normal.append(normal_right/(len(YValid)-sum(YValid).item()))
        accuracy_abnormal.append(abnormal_right/(sum(YValid).item()))
        print('accuracy_normal:%f' % (accuracy_normal[-1]))
        print('accuracy_abnormal:%f' % (accuracy_abnormal[-1]))

        if max_accuracy < temp:
            torch.save(Model_CNN.state_dict(), './'+model_dir+'/epoch' + str(epoch) + '_'
                       + str(accuracy[-1])[0:7] + '_' +
                       str((normal_right/(len(YValid)-sum(YValid))).numpy())[1:7] + '_'
                       + str((abnormal_right/(sum(YValid))).numpy())[1:7] + '.pth')
            max_accuracy = temp
        if epoch % 20 == 0:
            torch.save(Model_CNN.state_dict(), './'+model_dir+'/epoch' + str(epoch) + '_'
                       + str(accuracy[-1])[0:7] + '_' +
                       str((normal_right/(len(YValid)-sum(YValid))).numpy())[1:7] + '_'
                       + str((abnormal_right/(sum(YValid))).numpy())[1:7] + '.pth')
        elapsed = time.time() - start


        print("epoch:", epoch, "time used:", elapsed)
        save_metrics_to_excel(accuracy, accuracy_normal, accuracy_abnormal, model_dir)
#%%
    np.save('./'+model_dir+'/accuracy', accuracy)

#%%
    np.save('./'+model_dir+'/accuracy_normal', accuracy_normal)
#%%
    np.save('./'+model_dir+'/accuracy_abnormal', accuracy_abnormal)


