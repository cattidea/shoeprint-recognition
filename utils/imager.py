import os

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2 # 解决 pylint 中的 Error
from PIL import Image, ImageChops

# W = 80
# H = 208

# W = 45
# H = 117

W = 48
H = 132


density_noise = 0.01
num_block = 40
block_size = (30, 30)

def image2array(img_path, amplify=0):
    """ 将图像转化为向量，可扩增 """

    arrays = []
    imgs = []
    origin = Image.open(img_path).convert('1')

    if amplify:
        imgs += [origin]
        imgs += _transpose_amplify(imgs)
        imgs += _rotate_amplify(imgs, 5, -5) + \
                _offest_amplify(imgs, (30, 0), (-30, 0), (0, 30), (0, -30)) + \
                _noise_amplify(imgs, density_noise=density_noise) + \
                _random_block_amplify(imgs, num_block=num_block, block_size=block_size) + \
                _area_block_amplify(imgs)
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
def _random_block_amplify(img, num_block, block_size):
    """ 随机遮挡扩增 """
    w, h = img.size
    block_arr = np.array(img)
    thumbnail_mask = np.random.randint(0, num_block, ((h//block_size[0], w//block_size[1]))) > 1
    block_mask = np.array(Image.fromarray(thumbnail_mask, "L").resize((w, h)), dtype=np.bool_)
    block_arr &= block_mask
    return [Image.fromarray(block_arr, "L")]


@amplify
def _area_block_amplify(img):
    """ 区域遮挡扩增 """
    _, h = img.size
    block_arr = np.array(img)
    x_start = np.random.randint(h//3, h//3*2)
    x_end = np.random.randint(x_start, min(x_start+h//5, h))
    block_arr[x_start: x_end] = 0
    return [Image.fromarray(block_arr, "L")]


@amplify
def _noise_amplify(img, density_noise):
    """ 椒盐噪声扩增 """
    w, h = img.size
    noise_arr = np.array(img)
    noise_white_mask = np.random.rand(h,w) < density_noise
    noise_black_mask = np.random.rand(h,w) > density_noise
    noise_arr |= noise_white_mask
    noise_arr &= noise_black_mask
    return [Image.fromarray(noise_arr, "L")]


@amplify
def _offest_amplify(img, *offsets):
    """ 平移扩增 """
    return [ImageChops.offset(img, off_x, off_y) for off_x, off_y in offsets]


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
