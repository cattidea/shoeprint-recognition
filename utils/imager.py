import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# W = 80
# H = 208

W = 45
H = 117
density_noise = 0.01
num_block = 40
block_size = (30, 30)

def image2array(img_path, amplify=0):
    """ 将图像转化为向量，可传入旋转、镜像参数对数据进行扩增"""

    arrays = []
    imgs = []
    origin = Image.open(img_path).convert('1')

    if amplify:
        imgs = [origin]
        imgs = _transpose_amplify(imgs)
        imgs = _rotate_amplify(imgs, 5, -5)
        imgs = _noise_amplify(imgs, density_noise=density_noise)
        imgs = _block_amplify(imgs, num_block=num_block, block_size=block_size)
    else:
        imgs.append(origin)

    for img in imgs:
        arr = np.array(img.resize((W, H)), dtype=np.bool_).reshape((H, W, 1))
        arrays.append(arr)

    assert len(arrays) >= amplify
    return arrays


def amplify(func):
    """ 扩增装饰器，在原图片列表的基础上进行扩增 """
    def new_func(ori_imgs, *args, **kw):
        imgs = []
        for img in ori_imgs:
            imgs.append(img)
            imgs.extend(func(img, *args, **kw))
        return imgs
    return new_func

@amplify
def _rotate_amplify(img, *angles):
    """ 旋转扩增，可同时扩增多个角度 """
    return [img.rotate(angle) for angle in angles]


@amplify
def _transpose_amplify(img):
    """ 对称扩增 """
    return [img.transpose(Image.FLIP_LEFT_RIGHT)]


@amplify
def _block_amplify(img, num_block, block_size):
    """ 随机遮挡扩增 """
    w, h = img.size
    block_arr = np.array(img)
    for _ in range(num_block):
        x_start = np.random.randint(0, h-block_size[0])
        y_start = np.random.randint(0, w-block_size[0])
        block_arr[x_start: x_start+block_size[0], y_start: y_start+block_size[1]] = 0
    return [Image.fromarray(block_arr, "L")]


@amplify
def _noise_amplify(img, density_noise):
    """ 椒盐噪声扩增 """
    w, h = img.size
    noise_arr = np.array(img)
    for _ in range(int(density_noise*w*h)):
        noise_arr[np.random.randint(0, h), np.random.randint(0, w)] = np.random.randint(2)
    return [Image.fromarray(noise_arr, "L")]


def image2array_v2(img_path, amplify=False):
    """ opencv 接口的 image2array ， 注意路径名不能包含中文
    opencv 有着更高的性能（速度为 PIL 的两倍），但是变换的效果没有 PIL 理想（失真较严重）
    当前已弃用
    """

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
