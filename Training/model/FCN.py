# -*- coding: utf-8 -*-
"""
@Project: PythonDemo
@File   : FCN.py
@Author : Zhang P.H
@Date   : 2021/11/6
@Desc   :
"""
from torch import nn


class CNNDemo(nn.Module):
    def __init__(self):
        super(CNNDemo, self).__init__()
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 30), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(30, 10)
        )

    # input:(bt, 28, 28)
    def forward(self, x):
        return self.module(x)
