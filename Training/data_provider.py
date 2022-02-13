# -*- coding: utf-8 -*-
"""
@Project: 手写数字识别Demo
@File   : data_provider.py
@Author : Zhang P.H
@Date   : 2022/2/13
@Desc   :
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, data_dir):
        self.__data_dir = data_dir
        self.__image_list = os.listdir(os.path.join(self.__data_dir, "image"))
        self.__label_list = os.listdir(os.path.join(self.__data_dir, "label"))

    def __getitem__(self, item):
        image_path = os.path.join(self.__data_dir, "image", self.__image_list[item])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = (image/255).astype(np.float32)
        # print(image.dtype)
        label_path = os.path.join(self.__data_dir, "label", self.__label_list[item])
        label = np.loadtxt(label_path).astype(np.float32)
        return image, label

    def __len__(self):
        return len(self.__image_list)


if __name__ == '__main__':
    dataset = MyDataSet("../DataCollection/dataset/train")
    print(type(dataset[0][1]))
    cv2.imshow("", dataset[0][0])
    cv2.waitKey(10000)