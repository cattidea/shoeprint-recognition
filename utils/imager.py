import numpy as np
from PIL import Image

W = 30
H = 78

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
