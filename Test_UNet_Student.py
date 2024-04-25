#%%
import os
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
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
from matplotlib import pyplot as plt
from unet_student import UNetCustom
# from unet_at import UNetCustom
from unet_casd import UNetCustom

def calculate_miou(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    mIoU = np.nanmean(IoU)  # 计算平均 IoU，处理除以零的情况
    return mIoU

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


# 初始化模型
ndf = 18
Model_CNN = UNetCustom(1, ndf=ndf).cuda()
model_path = './train_unet_kd_casd/epoch361_0.9705.pth'
state_dict = torch.load(model_path)
# 删除不是模型参数的键
state_dict.pop('total_ops', None)  # 删除 'total_ops'，如果存在
state_dict.pop('total_params', None)  # 删除 'total_params'，如果存在
Model_CNN.load_state_dict(state_dict)
#%%

Model_CNN.eval()
#%%
predictions = []
true_labels = []

for batch_x2, batch_y2 in loader2:
    x2 = batch_x2.cuda()
    with torch.no_grad():
        outputs = Model_CNN(x2)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(batch_y2.view(-1).cpu().numpy())

# 计算召回率、精确度和F1分数
precision = precision_score(true_labels, predictions, average='binary')
recall = recall_score(true_labels, predictions, average='binary')
f1 = f1_score(true_labels, predictions, average='binary')

# 打印评估结果
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))
#%%
print(Model_CNN)

summary(Model_CNN, (1, XTrain.shape[2], XTrain.shape[3]))  # 65 33
input_t = torch.randn(1, 1, XTrain.shape[2], XTrain.shape[3]).cuda()
flops, params = profile(Model_CNN, inputs=(input_t,))
print(f"模型浮点运算次数（FLOPs）: {flops}")
print(f"模型参数数量（Params）: {params}")

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

# 计算 mIoU
miou = calculate_miou(true_labels, predictions)
print(f"平均交并比（Mean IoU）: {miou:.4f}")
print('accuracy_normal:%f' % (accuracy_normal))
print('accuracy_abnormal:%f' % (accuracy_abnormal))

