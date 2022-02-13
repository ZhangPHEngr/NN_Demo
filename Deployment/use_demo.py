# -*- coding: utf-8 -*-
"""
@Project: 手写数字识别Demo
@File   : use_demo.py
@Author : Zhang P.H
@Date   : 2022/2/13
@Desc   :
"""
import os
import sys

import cv2
import torch
import torch.onnx
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_PATH, "../"))

from Training.model.FCN import *

# 加载模型及参数
my_model = CNNDemo()
model_data = torch.load('my_model.pth')
my_model.load_state_dict(model_data)
print(my_model)

# 加载图像
my_model.eval()
image_path = "../DataCollection/personal/test1.png"
# image_path = "../DataCollection/data/dataset/test/image/000001.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = 255 - image  # 像素取反
image = cv2.resize(image,(28,28))  # 改成模型所需输入
cv2.imshow(" ", image)
cv2.waitKey(1000)
image = (image/255).astype(np.float32)
image = np.expand_dims(image, axis=0)
image = torch.from_numpy(image)
res = my_model(image)
print(res)
res = res.argmax(dim=1)
print("推断结果为：{}".format(res))

if __name__ == '__main__':
    pass
