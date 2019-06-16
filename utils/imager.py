import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# W = 80
# H = 208

W = 45
H = 117

def image2array(img_path, amplify=False):
    """ 将图像转化为向量，可传入旋转、镜像参数对数据进行扩增"""
    arrays = []
    imgs = []
    origin = Image.open(img_path).convert('1')

    if amplify:
        mirror = origin.transpose(Image.FLIP_LEFT_RIGHT)
        rotate_origin_01 = origin.rotate(5)
        rotate_origin_02 = origin.rotate(-5)
        rotate_mirror_01 = mirror.rotate(5)
        rotate_mirror_02 = mirror.rotate(-5)
        imgs.extend([origin, mirror, rotate_origin_01, rotate_origin_02, rotate_mirror_01, rotate_mirror_02])
    else:
        imgs.append(origin)

    for img in imgs:
        arr = np.array(img.resize((W, H)), dtype=np.bool_).reshape((H, W, 1))
        arrays.append(arr)
    return arrays

def image2array_v2(img_path, amplify=False):
    """ opencv 接口的 image2array ， 注意路径名不能包含中文
    opencv 有着更高的性能（速度为 PIL 的两倍），但是变换的效果没有 PIL 理想（失真较严重）"""

    arrays = []
    imgs = []
    img_path = os.path.normpath(img_path)
    origin = cv2.imread(img_path)
    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    if amplify:
        mirror = cv2.flip(origin, 1)
        w = origin.shape[1]
        h = origin.shape[0]
        center = (w//2, h//2)
        rotate_origin_01 = cv2.warpAffine(origin, cv2.getRotationMatrix2D(center, 5, 1), (w, h))
        rotate_origin_02 = cv2.warpAffine(origin, cv2.getRotationMatrix2D(center, -5, 1), (w, h))
        rotate_mirror_01 = cv2.warpAffine(mirror, cv2.getRotationMatrix2D(center, 5, 1), (w, h))
        rotate_mirror_02 = cv2.warpAffine(mirror, cv2.getRotationMatrix2D(center, -5, 1), (w, h))
        imgs.extend([origin, mirror, rotate_origin_01, rotate_origin_02, rotate_mirror_01, rotate_mirror_02])
    else:
        imgs.append(origin)

    for img in imgs:
        img = cv2.resize(img, (W, H))
        arr = np.array(img, dtype=np.bool_).reshape((H, W, 1))
        arrays.append(arr)
    return arrays

def plot(img_array):
    plt.imshow(np.reshape(img_array, (H, W)), cmap='gray')
    plt.show()
