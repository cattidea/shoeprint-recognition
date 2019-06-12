import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# W = 80
# H = 208

W = 45
H = 117

def image2array(img_path, rotate=False, transpose=False):
    """ 将图像转化为向量，可传入旋转、镜像参数对数据进行扩增"""
    arrays = []
    im = Image.open(img_path).convert('1')
    imgs = [im]
    if rotate:
        imgs.append(im.rotate(5))
        imgs.append(im.rotate(-5))
    if transpose:
        imgs.append(im.transpose(Image.FLIP_LEFT_RIGHT))
    for img in imgs:
        arr = np.array(img.resize((W, H)), dtype=np.bool_).reshape((H, W, 1))
        arrays.append(arr)
    return arrays

def image2array_v2(img_path, rotate=False, transpose=False):
    """ opencv 接口的 image2array ， 注意路径名不能包含中文
    opencv 有着更高的性能（速度为 PIL 的两倍），但是变换的效果没有 PIL 理想（失真较严重）"""

    arrays = []
    img_path = os.path.normpath(img_path)
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgs = [im]
    if rotate:
        center = (im.shape[1]//2, im.shape[0]//2)
        rotate_matrix_1 = cv2.getRotationMatrix2D(center, 5, 1)
        rotate_matrix_2 = cv2.getRotationMatrix2D(center, -5, 1)
        imgs.append(cv2.warpAffine(img, rotate_matrix_1, center))
        imgs.append(cv2.warpAffine(img, rotate_matrix_2, center))
    if transpose:
        imgs.append(cv2.flip(im, 1))
    for img in imgs:
        img = cv2.resize(img, (W, H))
        arr = np.array(img, dtype=np.bool_).reshape((H, W, 1))
        arrays.append(arr)
    return arrays

def plot(img_array):
    plt.imshow(np.reshape(img_array, (H, W)), cmap='gray')
    plt.show()
