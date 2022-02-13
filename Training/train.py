# -*- coding: utf-8 -*-
"""
@Project: 手写数字识别Demo
@File   : train.py
@Author : Zhang P.H
@Date   : 2022/2/13
@Desc   :
"""
import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from data_provider import *
from model.FCN import *

# 0.全局设置
BATCH_SIZE = 128
EPOCH = 150
learning_rate = 0.1

# 1.准备数据
train_dataset = MyDataSet("../DataCollection/dataset/train")
test_dataset = MyDataSet("../DataCollection/dataset/test")
# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(train_dataset[0][0].shape)

# 2.准备网络
my_net = CNNDemo()
print("当前网络结构：\n", my_net)

# 3.设置损失函数
loss_fn = nn.CrossEntropyLoss()

# 4.设置优化器
optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate)

# ----------------------------------------开始训练--------------------------------------- #
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(EPOCH):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    print("-------训练模式-------")
    my_net.train()  # 设置模型为训练模式
    for train_data in train_dataloader:
        imgs, targets = train_data
        outputs = my_net(imgs)
        loss = loss_fn(outputs, targets.argmax(dim=1))

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    print("-------验证模式-------")
    my_net.eval()

    ap = 0
    total_test_step = 0
    for test_data in test_dataloader:
        imgs_test, targets_test = train_data
        outputs_test = my_net(imgs_test)  # （128，10)
        outputs_test = outputs_test.argmax(dim=1).numpy()
        targets_test = targets_test.argmax(dim=1).numpy()
        ap += np.sum((outputs_test == targets_test) != 0)
        total_test_step += 1
    ap /= total_test_step * BATCH_SIZE
    print("AP:{}".format(ap))

    if ap >= 0.99:
        break

# 输出模型参数
torch.save(my_net.state_dict(), "my_model.pth")

if __name__ == '__main__':
    pass
