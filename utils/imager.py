import numpy as np
from PIL import Image

W = 30
H = 78
GAMMA = 0.05

def image2array(img_path, rotate=False, transpose=False):
    """ 将图像转化为向量以及存在标签，可传入旋转、镜像参数对数据进行扩增"""
    arrays = []
    im = Image.open(img_path).convert('1')
    imgs = [im]
    if rotate:
        imgs.append(im.rotate(5))
        imgs.append(im.rotate(-5))
    if transpose:
        imgs.append(im.transpose(Image.FLIP_LEFT_RIGHT))
    for img in imgs:
        arr = np.array(img.resize([W, H]), dtype=np.bool_).reshape((H, W, 1))
        tag = get_tag(arr)
        arrays.append((arr, tag))
    return arrays

def get_tag(arr):
    """ 对上中下三部分分别打标签，如果该部分鞋印占比高于阈值 GAMMA 则标记为 True"""
    arr1 = arr[0: W]
    arr2 = arr[int((H-W)/2): int((H+W)/2)]
    arr3 = arr[H-W: H]
    arrs = [arr1, arr2, arr3]
    tag = np.zeros((3, ), dtype=np.bool_)

    found = False
    max_index = 0
    max_rate = -1
    for i in range(3):
        rate = np.sum(arrs[i]) / W**2
        if rate > max_rate:
            max_rate = rate
            max_index = i
        if rate > GAMMA:
            found = True
            tag[i] = True

    if not found:
        tag[max_index] = True

    return tag


