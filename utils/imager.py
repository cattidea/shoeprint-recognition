import os

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2 # 解决 pylint 中的 Error
from PIL import Image, ImageChops


W = 48
H = 132


def image2array(img_path, amplify=0):
    """ 将图像转化为向量，可扩增 """

    arrays = []
    img_arrs = []
    origin = _read_img_cv(img_path)

    if amplify:
        img_arrs += [origin]
        img_arrs += _transpose_amplify_cv(img_arrs)
        img_arrs += _rotate_amplify_cv(img_arrs, 10, -10) + \
                    _offest_amplify_cv(img_arrs, (40, 0), (-40, 0), (0, 40), (0, -40)) + \
                    _noise_amplify(img_arrs, density_noise=0.02) + \
                    _random_block_amplify(img_arrs, num_block=30, block_size=(40, 40)) + \
                    _area_block_amplify(img_arrs)
    else:
        img_arrs.append(origin)

    for img_arr in img_arrs:
        arr = _resize_cv(img_arr, (W, H)).reshape((H, W, 1))
        arrays.append(arr)

    assert len(arrays) >= amplify
    return arrays


def _read_img(img_path):
    """ 读取图片 """
    return np.array(Image.open(img_path).convert('L'), dtype=np.uint8)

def _read_img_cv(img_path):
    """ 读取图片 opencv 版 """
    img_path = os.path.normpath(img_path)
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


def _resize(img_arr, shape):
    """ 图片矩阵改变大小 """
    return np.array(Image.fromarray(img_arr, "L").resize(shape), dtype=np.uint8)


def _resize_cv(img_arr, shape):
    """ 图片矩阵改变大小 opencv 版 """
    return cv2.resize(img_arr, shape)


def amplify(func):
    """ 扩增装饰器，在原图片列表的基础上进行扩增 """
    def new_func(ori_img_arrs, *args, **kw):
        img_arrs = []
        for img_arr in ori_img_arrs:
            img_arrs.extend(func(img_arr, *args, **kw))
        return img_arrs
    return new_func


@amplify
def _transpose_amplify(img_arr):
    """ 对称扩增 """
    img = Image.fromarray(img_arr, "L")
    return [np.asarray(img.transpose(Image.FLIP_LEFT_RIGHT))]


@amplify
def _transpose_amplify_cv(img_arr):
    """ 对称扩增 opencv 版 """
    return [cv2.flip(img_arr, 1)]


@amplify
def _rotate_amplify(img_arr, *angles):
    """ 旋转扩增，可同时扩增多个角度 """
    img = Image.fromarray(img_arr, "L")
    return [np.asarray(img.rotate(angle)) for angle in angles]


@amplify
def _rotate_amplify_cv(img_arr, *angles):
    """ 旋转扩增 opencv 版 """
    h, w = img_arr.shape
    center = (w//2, h//2)
    return [cv2.warpAffine(img_arr, cv2.getRotationMatrix2D(center, angle, 1), (w, h)) for angle in angles]


@amplify
def _offest_amplify(img_arr, *offsets):
    """ 平移扩增 """
    img = Image.fromarray(img_arr, "L")
    return [np.asarray(ImageChops.offset(img, off_x, off_y)) for off_x, off_y in offsets]


@amplify
def _offest_amplify_cv(img_arr, *offsets):
    """ 平移扩增 opencv 版 """
    h, w = img_arr.shape
    return [cv2.warpAffine(img_arr, np.float32([[1,0,off_x],[0,1,off_y]]), (w, h)) for off_x, off_y in offsets]


@amplify
def _noise_amplify(img_arr, density_noise):
    """ 椒盐噪声扩增 """
    h, w = img_arr.shape
    noise_arr = img_arr.copy()
    noise_white_mask = np.random.rand(h,w) < density_noise
    noise_black_mask = np.random.rand(h,w) > density_noise
    noise_white_mask = noise_white_mask.astype(np.uint8) * 255
    noise_black_mask = noise_black_mask.astype(np.uint8) * 255
    noise_arr |= noise_white_mask
    noise_arr &= noise_black_mask
    return [noise_arr]


@amplify
def _random_block_amplify(img_arr, num_block, block_size):
    """ 随机遮挡扩增 """
    h, w = img_arr.shape
    block_arr = img_arr.copy()
    thumbnail_mask = np.random.randint(0, num_block, ((h//block_size[0], w//block_size[1]))) > 1
    block_mask = _resize_cv(thumbnail_mask.astype(np.uint8), (w, h)) * 255
    block_arr &= block_mask
    return [block_arr]


@amplify
def _area_block_amplify(img_arr):
    """ 区域遮挡扩增 """
    h, _ = img_arr.shape
    block_arr = img_arr.copy()
    x_start = np.random.randint(h//3, h//3*2)
    x_end = np.random.randint(x_start, min(x_start+h//5, h))
    block_arr[x_start: x_end] = 0
    return [block_arr]


@amplify
def _deformation_amplify(img_arr, kernel_size=(3, 3)):
    """ 弹性形变扩增
    有问题，且太慢，暂不启用 """
    h, w = img_arr.shape
    kernel = np.random.normal(0, 0, kernel_size)
    matrix_h = np.random.randint(10, size=(h, w))
    matrix_v = np.random.randint(10, size=(h, w))
    matrix_h = conv2(matrix_h, kernel)
    matrix_v = conv2(matrix_v, kernel)
    defor_arr = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            new_i = i + int(matrix_v[i][j])
            new_j = j + int(matrix_h[i][j])
            if new_i in range(h) and new_j in range(w):
                defor_arr[new_i][new_j] = img_arr[i][j]
    return [defor_arr]


def conv2(img, kernel):
    """ 对图片矩阵进行卷积处理
    ref: https://blog.csdn.net/u013044310/article/details/82786957
    """
    H, W = img.shape
    kernel_size = kernel.shape
    col = np.zeros((H, kernel_size[1]//2))
    row = np.zeros((kernel_size[0]//2, W + kernel_size[1]//2 * 2))
    img = np.insert(img, [W], values=col, axis=1)
    img = np.insert(img, [0], values=col, axis=1)
    img = np.insert(img, H, values=row, axis=0)
    img = np.insert(img, 0, values=row, axis=0)
    res = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            temp = img[i:i + kernel_size[0], j:j + kernel_size[1]]
            temp = np.multiply(temp, kernel)
            res[i][j] = temp.sum()
    return res


def plot(img_array):
    plt.imshow(np.reshape(img_array, (H, W)), cmap='gray')
    plt.show()
