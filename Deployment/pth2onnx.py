# -*- coding: utf-8 -*-
"""
@Project: 手写数字识别Demo
@File   : pth2onnx.py
@Author : Zhang P.H
@Date   : 2022/2/13
@Desc   :
"""
import os
import sys

import torch
import torch.onnx

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_PATH, "../"))

from Training.model.FCN import *

# 加载模型及参数
my_model = CNNDemo()
model_data = torch.load('my_model.pth')
my_model.load_state_dict(model_data)
print(my_model)

# 转为onnx
my_model.eval()
batch_size = 1
x = torch.randn(batch_size, 28, 28, requires_grad=True)
torch_out = my_model(x)

torch.onnx.export(my_model,                 # model
                  x,                        # model input (or a tuple for multiple inputs)
                  "mnist.onnx",             # where to save the model
                  export_params=True,       # store the trained parameter weights inside the model file
                  opset_version=10,         # the ONNX version to export the model to
                  do_constant_folding=True, # whether to execute constant folding for optimization
                  input_names=['input'],    # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={
                      'input': [batch_size],
                      'output': [batch_size]
                  })

if __name__ == '__main__':
    pass
