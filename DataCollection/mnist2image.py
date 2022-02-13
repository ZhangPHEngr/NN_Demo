# -*- coding: utf-8 -*-
"""
@Project: 手写数字识别Demo
@File   : mnist2image.py
@Author : Zhang P.H
@Date   : 2022/2/13
@Desc   : https://blog.csdn.net/weixin_40522523/article/details/82823812
"""
import os
from tqdm import tqdm
import cv2
import numpy as np
from struct import unpack


def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(train_image_path, train_label_path, test_image_path, test_label_path, normalize=True, one_hot=True):
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train': __read_image(train_image_path),
        'test': __read_image(test_image_path)
    }

    label = {
        'train': __read_label(train_label_path),
        'test': __read_label(test_label_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])


def save(image, label, path):
    os.makedirs(os.path.join(path, "image"),exist_ok=True)
    os.makedirs(os.path.join(path, "label"), exist_ok=True)
    cnt = 0
    # 写入图像
    bar = tqdm(image)
    for item in bar:
        bar.set_description("处理到{}".format(cnt))
        item = item.reshape(28, 28)
        # print(item)
        image_file = os.path.join(path, "image/%06d" % cnt + ".jpg")
        cv2.imwrite(image_file, item)
        label_file = image_file.replace("image", "label")
        label_file = label_file.replace(".jpg", ".txt")
        np.savetxt(label_file, label[cnt].reshape(1, -1), fmt='%d')
        cnt += 1


if __name__ == '__main__':
    train_image_path = "dataset/original/train-images.idx3-ubyte"
    train_label_path = "dataset/original/train-labels.idx1-ubyte"
    test_image_path = "dataset/original/t10k-images.idx3-ubyte"
    test_label_path = "dataset/original/t10k-labels.idx1-ubyte"
    (train_image, train_label), (test_image, test_label) = load_mnist(train_image_path, train_label_path,
                                                                      test_image_path, test_label_path, normalize=False)
    print("训练集大小：{} 测试集大小：{} 目标类别数：{}".format(train_label.shape[0], test_label.shape[0], test_label.shape[1]))
    save(train_image, train_label, "dataset/train")
    save(test_image, test_label, "dataset/test")
