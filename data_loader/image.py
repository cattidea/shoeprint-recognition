import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

from config_parser.config import CONFIG, IMAGE_PARAMS

try:
    import deformation
    HAS_DEFORMATION_LIB = True
except ModuleNotFoundError:
    HAS_DEFORMATION_LIB = False


W = IMAGE_PARAMS["W"]
H = IMAGE_PARAMS["H"]

TRANSPOSE = lambda x: _transpose_augment_cv(x)
ROTATE = lambda x: _rotate_augment_cv(x, 10, -10)
OFFSET = lambda x: _offest_augment_cv(x, (40, 0), (-40, 0), (0, 40), (0, -40))
NOISE = lambda x: _noise_augment(x, density_noise=0.02)
RANDOM_BLOCK = lambda x: _random_block_augment(x, num_block=30, block_size=(40, 40))
AREA_BLOCK = lambda x: _area_block_augment(x)
DEFORMATION = lambda x: _deformation_augment(x, k=500, kernel_size=(225, 225), sigma=15)
MEDIAN_BLUR = lambda x: _median_blur_augment(x)
GAUSSIAN_BLUR = lambda x: _gaussian_blur_augment(x)
BILATERAL_BLUR = lambda x: _bilateral_blur_augment(x)
AVERAGE_BLUR = lambda x: _average_blur_augment(x)
ALL = [TRANSPOSE, (ROTATE, OFFSET, NOISE, RANDOM_BLOCK, AREA_BLOCK, DEFORMATION)]


def image2array(img_path, augment=ALL):
    """ 将图像转化为向量，可扩增
    augment: list<method or tuple<method>> """

    arrays = []
    origin = _read_img_cv(img_path)
    img_arrs = [origin]

    if augment:
        for mets in augment:
            if isinstance(mets, tuple):
                img_arrs_tmp = []
                for method in mets:
                    img_arrs_tmp += method(img_arrs)
                img_arrs += img_arrs_tmp
            else:
                img_arrs += mets(img_arrs)

    for img_arr in img_arrs:
        arr = _resize_cv(img_arr, (W, H)).reshape((H, W, 1))
        arrays.append(arr)

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


def _resize_cv(img_arr, shape, interpolation=cv2.INTER_LINEAR):
    """ 图片矩阵改变大小 opencv 版 """
    return cv2.resize(img_arr, shape, interpolation=cv2.INTER_LINEAR)


def augment(func):
    """ 扩增装饰器，在原图片列表的基础上进行扩增 """
    def new_func(ori_img_arrs, *args, **kw):
        img_arrs = []
        for img_arr in ori_img_arrs:
            img_arrs.extend(func(img_arr, *args, **kw))
        return img_arrs
    return new_func


@augment
def _transpose_augment(img_arr):
    """ 对称扩增 """
    img = Image.fromarray(img_arr, "L")
    return [np.asarray(img.transpose(Image.FLIP_LEFT_RIGHT))]


@augment
def _transpose_augment_cv(img_arr):
    """ 对称扩增 opencv 版 """
    return [cv2.flip(img_arr, 1)]


@augment
def _rotate_augment(img_arr, *angles):
    """ 旋转扩增，可同时扩增多个角度 """
    img = Image.fromarray(img_arr, "L")
    return [np.asarray(img.rotate(angle)) for angle in angles]


@augment
def _rotate_augment_cv(img_arr, *angles):
    """ 旋转扩增 opencv 版 """
    h, w = img_arr.shape
    center = (w//2, h//2)
    return [cv2.warpAffine(img_arr, cv2.getRotationMatrix2D(center, angle, 1), (w, h)) for angle in angles]


@augment
def _offest_augment(img_arr, *offsets):
    """ 平移扩增 """
    img = Image.fromarray(img_arr, "L")
    return [np.asarray(ImageChops.offset(img, off_x, off_y)) for off_x, off_y in offsets]


@augment
def _offest_augment_cv(img_arr, *offsets):
    """ 平移扩增 opencv 版 """
    h, w = img_arr.shape
    return [cv2.warpAffine(img_arr, np.float32([[1, 0, off_x], [0, 1, off_y]]), (w, h)) for off_x, off_y in offsets]


@augment
def _noise_augment(img_arr, density_noise):
    """ 椒盐噪声扩增 """
    h, w = img_arr.shape
    noise_arr = img_arr.copy()
    noise_white_mask = np.random.rand(h, w) < density_noise
    noise_black_mask = np.random.rand(h, w) > density_noise
    noise_white_mask = noise_white_mask.astype(np.uint8) * 255
    noise_black_mask = noise_black_mask.astype(np.uint8) * 255
    noise_arr |= noise_white_mask
    noise_arr &= noise_black_mask
    return [noise_arr]


@augment
def _random_block_augment(img_arr, num_block, block_size):
    """ 随机遮挡扩增 """
    h, w = img_arr.shape
    block_arr = img_arr.copy()
    thumbnail_mask = np.random.randint(
        0, num_block, ((h//block_size[0], w//block_size[1]))) > 1
    block_mask = _resize_cv(thumbnail_mask.astype(np.uint8), (w, h)) * 255
    block_arr &= block_mask
    return [block_arr]


@augment
def _area_block_augment(img_arr):
    """ 区域遮挡扩增 """
    h, _ = img_arr.shape
    block_arr = img_arr.copy()
    x_start = np.random.randint(h//3, h//3*2)
    x_end = np.random.randint(x_start, min(x_start+h//5, h))
    block_arr[x_start: x_end] = 0
    return [block_arr]


@augment
def _deformation_augment(img_arr, k=500, kernel_size=(225, 225), sigma=15):
    """ 弹性形变扩增 """
    h, w = img_arr.shape
    matrix_h = np.random.uniform(low=-1.0*k, high=1.0*k, size=(h, w))
    matrix_v = np.random.uniform(low=-1.0*k, high=1.0*k, size=(h, w))
    matrix_h = cv2.GaussianBlur(matrix_h, kernel_size, sigma).astype(np.int32)
    matrix_v = cv2.GaussianBlur(matrix_v, kernel_size, sigma).astype(np.int32)
    defor_arr = np.zeros((h, w), dtype=np.uint8)

    if HAS_DEFORMATION_LIB:
        deformation.deformation_change_matrix(defor_arr, img_arr, matrix_h ,matrix_v)
    else:
        for i in range(h):
            for j in range(w):
                new_i = i + int(matrix_v[i][j])
                new_j = j + int(matrix_h[i][j])
                if new_i in range(h) and new_j in range(w):
                    defor_arr[new_i][new_j] = img_arr[i][j]
    return [defor_arr]


@augment
def _deformation_augment_v2(img_arr, k=500, kernel_size=(225, 225), sigma=15):
    """ 弹性形变扩增 v2 ，暂不可用 """
    h, w = img_arr.shape
    matrix_h = np.random.uniform(low=-1.0*k, high=1.0*k, size=(h, w))
    matrix_v = np.random.uniform(low=-1.0*k, high=1.0*k, size=(h, w))
    matrix_h = cv2.GaussianBlur(matrix_h, kernel_size, sigma).astype(np.int32)
    matrix_v = cv2.GaussianBlur(matrix_v, kernel_size, sigma).astype(np.int32)
    matrix_x = np.concatenate([np.arange(0, h).reshape((h, 1))
                               for _ in range(w)], axis=1)
    matrix_y = np.concatenate([np.arange(0, w).reshape((1, w))
                               for _ in range(h)], axis=0)
    matrix_x += matrix_v
    matrix_y += matrix_h
    matrix_x = np.where(matrix_x > 0, matrix_x, 0)
    matrix_x = np.where(matrix_x < h, matrix_x, h)
    matrix_y = np.where(matrix_y > 0, matrix_y, 0)
    matrix_y = np.where(matrix_y < w, matrix_y, w)
    img_triplet = np.stack([matrix_x, matrix_y, img_arr],
                           axis=-1).reshape((w*h, 3))
    img_triplet = np.array(sorted(img_triplet, key=lambda t: t[0]*10000+t[1]))
    defor_arr = img_triplet.reshape((h, w, 3))[:, :, 2].astype(np.uint8)
    return [defor_arr]


@augment
def _median_blur_augment(img_arr):
    """ 中值滤波扩增 """
    return [cv2.medianBlur(img_arr, 5)]


@augment
def _gaussian_blur_augment(img_arr):
    """ 高斯滤波扩增 """
    return [cv2.GaussianBlur(img_arr, (7, 7), 0)]


@augment
def _bilateral_blur_augment(img_arr):
    """ 双边滤波扩增 """
    return [cv2.bilateralFilter(img_arr, 40, 75, 75)]


@augment
def _average_blur_augment(img_arr):
    """ 均值滤波扩增 """
    return [cv2.blur(img_arr, (5, 5))]


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


def img_plot(*img_arrays, shape=(H, W)):
    """ 每张图片 reshape，之后同时并列显示多张图片 """
    img_arrays = [np.reshape(img_array, shape) for img_array in img_arrays]
    img = np.concatenate(img_arrays, axis=1)
    plt.imshow(img, cmap='gray')
    plt.show()
