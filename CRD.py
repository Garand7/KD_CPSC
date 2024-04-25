import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pandas as pd
from torch import optim

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
from sklearn.metrics import accuracy_score
from unet_at import UNetCustom as UNet_s
from unet_teacher import UNetCustom as UNet_t
start = time.time()
np.random.seed(1337)

ndf = 18
alpha = 0.01
XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)
elapsed = time.time() - start
print("data load time used:", elapsed)

def CRD(t_logits, s_logits, temperature):
    T = temperature
    # Use cosine similarity as the distance metric
    t_logits = F.normalize(t_logits.view(t_logits.size(0), -1), dim=-1).float()
    s_logits = F.normalize(s_logits.view(s_logits.size(0), -1), dim=-1).float()

    N, C = t_logits.size()[0], t_logits.size()[1]

    # calculate teacher similarity matrix
    t_similarity_matrix = torch.matmul(t_logits, t_logits.t()) / T
    t_similarity_matrix = torch.softmax(t_similarity_matrix, dim=-1)
    # dimension reduction on teacher similarity matrix
    c_t = torch.mean(t_similarity_matrix, dim=-1)

    # calculate student similarity matrix
    s_similarity_matrix = torch.matmul(s_logits, s_logits.t()) / T
    s_similarity_matrix = torch.softmax(s_similarity_matrix, dim=-1)
    # dimension reduction on student similarity matrix
    c_s = torch.mean(s_similarity_matrix, dim=-1)

    # calculate contrastive loss
    loss = (c_t - c_s).pow(2).mean()

    return loss



# 定义蒸馏损失
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=5):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # 确保标签是1D的
        labels = labels.squeeze()

        # 计算软目标损失
        soft_logits = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_logits, soft_labels.detach(), reduction='batchmean') * (self.temperature ** 2)

        # 计算真实标签损失
        label_loss = self.criterion(student_logits, labels)

        return (1 - self.alpha) * label_loss + self.alpha * distillation_loss

# 初始化教师和学生网络
teacher_model = UNet_t(1, ndf).cuda()
student_model = UNet_s(1, ndf).cuda()


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

# 加载教师网络权重

model_path = './train_unet_teacher_18_0.010000_00/epoch55_0.97206_0.9741_0.9604.pth'
state_dict = torch.load(model_path)
# 删除不是模型参数的键
state_dict.pop('total_ops', None)  # 删除 'total_ops'，如果存在
state_dict.pop('total_params', None)  # 删除 'total_params'，如果存在
teacher_model.load_state_dict(state_dict)
teacher_model.eval()

#%%
# 设置学生网络的训练环境
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
distill_criterion = DistillationLoss(alpha=0.5, temperature=3)


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
    model_dir = 'train_unet_kd_crd'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    max_accuracy = 0  # Track the best validation accuracy
    loss_list = []
    accuracy_list = []

    for epoch in range(201):
        student_model.train()
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda().squeeze()

            with torch.no_grad():
                teacher_outputs = teacher_model(batch_x)

            student_outputs = student_model(batch_x)
            loss_ce = distill_criterion(student_outputs, teacher_outputs, batch_y)
            loss_crd = CRD(teacher_outputs,student_outputs,5)
            loss = 0.5*loss_ce+ 0.8*loss_crd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        loss_list.append(average_loss)

        # Calculate validation accuracy
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x2, batch_y2 in loader2:
                batch_x2, batch_y2 = batch_x2.cuda(), batch_y2.cuda().squeeze()
                outputs = student_model(batch_x2)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y2).sum().item()
                total += batch_y2.size(0)

        accuracy = correct / total
        accuracy_list.append(accuracy)

        print(f'Epoch {epoch}, Average Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.4f}')
        # Save model if it has the best accuracy so far or every 20 epochs
        if accuracy > max_accuracy or epoch % 20 == 0:
            model_path = os.path.join(model_dir, f'epoch{epoch}_{accuracy:.4f}.pth')
            torch.save(student_model.state_dict(), model_path)
            max_accuracy = max(max_accuracy, accuracy)

    save_metrics_to_excel(loss_list, accuracy_list, model_dir)

    # Save metrics to disk
    np.save(os.path.join(model_dir, 'loss_list.npy'), np.array(loss_list))
    np.save(os.path.join(model_dir, 'accuracy_list.npy'), np.array(accuracy_list))
