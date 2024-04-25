import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pandas as pd
from torch import optim

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
# from focalloss import FocalLoss
from fcalloss import FocalLoss
from torchsummary import summary
from thop import profile
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from unet_student import UNetCustom as UNet_s
from unet_teacher import UNetCustom as UNet_t
start = time.time()
np.random.seed(1337)

ndf = 18

XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)
elapsed = time.time() - start
print("data load time used:", elapsed)

def save_metrics_to_excel(loss_list, accuracy_list, model_dir):
    # Create a DataFrame with epoch, loss, and accuracy data
    df = pd.DataFrame({
        'Epoch': list(range(len(loss_list))),
        'Loss': loss_list,
        'Accuracy': accuracy_list
    })
    # Specify the save directory
    excel_path = os.path.join(model_dir, 'training_metrics.xlsx')
    # Save to an Excel file
    df.to_excel(excel_path, index=False)
    print(f"Metrics saved to {excel_path}")


# 初始化网络

model = UNet_s(1, ndf).cuda()


# 设置网络的训练环境
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


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

#%%
    model_dir = 'train_unet_student_ce2'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    max_accuracy = 0
    loss_list = []
    accuracy_list = []

    for epoch in range(201):
        model.train()
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda().squeeze()

            student_outputs = model(batch_x)
            loss = criterion(student_outputs, batch_y)
            # loss = FocalLoss(gamma=2, alpha=[0.01, 1])(student_outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        loss_list.append(average_loss)

        # Calculate validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x2, batch_y2 in loader2:
                batch_x2, batch_y2 = batch_x2.cuda(), batch_y2.cuda().squeeze()
                outputs = model(batch_x2)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y2).sum().item()
                total += batch_y2.size(0)

        accuracy = correct / total
        accuracy_list.append(accuracy)

        print(f'Epoch {epoch}, Average Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.4f}')
        # Save model if it has the best accuracy so far or every 20 epochs
        if accuracy > max_accuracy or epoch % 20 == 0:
            model_path = os.path.join(model_dir, f'epoch{epoch}_{accuracy:.4f}.pth')
            torch.save(model.state_dict(), model_path)
            max_accuracy = max(max_accuracy, accuracy)

    save_metrics_to_excel(loss_list, accuracy_list, model_dir)

    # Save metrics to disk
    np.save(os.path.join(model_dir, 'loss_list.npy'), np.array(loss_list))
    np.save(os.path.join(model_dir, 'accuracy_list.npy'), np.array(accuracy_list))
