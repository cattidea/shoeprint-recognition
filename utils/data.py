import os
import random

import h5py
import numpy as np

from utils.config import Config
from utils.imager import H, W, image2array

CONFIG = Config()
SIMPLE_DIR = CONFIG['simple_dir']
DETERMINE_FILE = CONFIG["determine_file"]
SHOEPRINT_DIR = CONFIG["shoeprint_dir"]
H5_PATH = CONFIG["h5_path"]


def get_simple_arrays(amplify):
    """ 获取样本文件结构，将样本图片预处理成所需格式
    ``` json
    {
        "type_num": {
            "imgs": [img1, img2, img3, ...],
        },
        ...
    }
    ```
    """
    rotate, transpose = bool(amplify), bool(amplify)
    simple_arrays = {}
    types = os.listdir(SIMPLE_DIR)
    for i, tp in enumerate(types):
        print("get_simple_arrays {}/{}".format(i, len(types)), end='\r')
        type_dir = os.path.join(SIMPLE_DIR, tp)
        img_path = os.path.join(type_dir, os.listdir(type_dir)[0])
        simple_arrays[tp] = {}
        simple_arrays[tp]["imgs"] = image2array(img_path, rotate, transpose)
    return simple_arrays


def get_shoeprint_arrays(amplify):
    """ 获取鞋印文件结构，将鞋印图片预处理成所需格式
    ``` json
    {
        "name": {
            "type_num": "xxxxxxxx",
            "imgs": [img1, img2, img3, ...],
        },
        ...
    }
    ```
    """
    rotate, transpose = bool(amplify), bool(amplify)
    shoeprint_arrays = {}
    types = os.listdir(SHOEPRINT_DIR)
    for i, tp in enumerate(types):
        print("get_shoeprint_arrays {}/{}".format(i, len(types)), end='\r')
        type_dir = os.path.join(SHOEPRINT_DIR, tp)
        for filename in os.listdir(type_dir):
            img_path = os.path.join(type_dir, filename)
            shoeprint_arrays[filename] = {}
            shoeprint_arrays[filename]["type_num"] = tp
            shoeprint_arrays[filename]["imgs"] = image2array(
                img_path, rotate, transpose)
    return shoeprint_arrays


def get_determine_scope():
    """ 读取待判定范围文件，并构造成字典型
    ``` json
    {
        "name": [
            P, N1, N2, N3, ... // 注意， P 不一定在最前面，而且这里记录的是 type_num
        ],
        ...
    }
    ```
    """
    determine_scope = {}
    with open(DETERMINE_FILE, 'r') as f:
        for line in f:
            line_items = line.split('\t')
            for i in range(len(line_items)):
                line_items[i] = line_items[i].strip()
            determine_scope[line_items[0]] = line_items[1:]
    return determine_scope


def get_img_three_tuples(amplify):
    """ 获取图片三元组， 可对数据进行扩增 amplify in (1, 9) ，最终结果是乱序的三元组数据
    ``` python
    [
        (
            (A_img, A_tag),
            (P_img, P_tag),
            (N_img, N_tag)
        ),
        ...
    ]
    ```
    """

    determine_scope = get_determine_scope()
    simple_arrays = get_simple_arrays(amplify)
    shoeprint_arrays = get_shoeprint_arrays(amplify)
    img_three_tuples = []

    for i, img_name in enumerate(determine_scope):
        print("get_img_three_tuples {}/{}".format(i, len(determine_scope)), end='\r')
        if img_name in shoeprint_arrays:
            positive_type_num = shoeprint_arrays[img_name]["type_num"]
            for negative_type_num in determine_scope[img_name]:
                if negative_type_num == positive_type_num:
                    continue

                img_three_tuple_list = [(a, p, n) for a in shoeprint_arrays[img_name]["imgs"]
                                        for p in simple_arrays[positive_type_num]["imgs"]
                                        for n in simple_arrays[negative_type_num]["imgs"]]

                if amplify:
                    img_three_tuple_list = random.sample(
                        img_three_tuple_list, 3*amplify)

                img_three_tuples.extend(img_three_tuple_list)
    random.shuffle(img_three_tuples)
    return img_three_tuples


def get_data_set(amplify):
    """ 将三元组数据转化为数据集格式
    ``` h5
    {
        "X": [
            [A_img, P_img, N_img], # 每个都是 (30, 84)
            ...
            ]
        "X_tag": [
            [A_tag, P_tag, N_tag], # 每个都是 (3, )
            ...
        ]
    }
    ```
    """

    img_three_tuples = get_img_three_tuples(amplify)
    length = len(img_three_tuples)
    data_set = {}

    X = np.zeros(shape=(length, 3, H, W), dtype=np.bool_)
    X_tag = np.zeros(shape=(length, 3, 3), dtype=np.bool_)

    for i in range(length):
        for j in range(3):
            X[i][j], X_tag[i][j] = img_three_tuples[i][j]

    data_set["X"] = X
    data_set["X_tag"] = X_tag
    random_divide(data_set)
    return data_set


def random_divide(data_set, proportion=(0.95, 0.05)):
    """ 将数据集切分为训练集与开发集 """
    length = len(data_set)
    indexes = list(np.random.permutation(length))
    train_size = round(proportion[0] * length)
    dev_size = round(proportion[1] * length)
    train_indexes = indexes[: train_size]
    dev_indexes = indexes[train_size:]

    # 初始化
    data_set["X_train_set"] = np.zeros(shape=(train_size, 3, H, W), dtype=np.bool_)
    data_set["X_tag_train_set"] = np.zeros(shape=(dev_size, 3, 3), dtype=np.bool_)
    data_set["X_dev_set"] = np.zeros(shape=(dev_size, 3, H, W), dtype=np.bool_)
    data_set["X_tag_dev_set"] = np.zeros(shape=(dev_size, 3, 3), dtype=np.bool_)

    # 更新数值
    data_set["X_train_set"], data_set["X_tag_train_set"] = data_set["X"][train_indexes], data_set["X_tag"][train_indexes]
    data_set["X_dev_set"], data_set["X_tag_dev_set"] = data_set["X"][dev_indexes], data_set["X_tag"][dev_indexes]


def data_import(amplify=0):
    """ 导入数据集， 分为训练集、开发集
    ``` h5
    {
        "X_train_set": [
            [A_img, P_img, N_img], # 每个都是 (30, 84)
            ...
            ]
        "X_tag_train_set": [
            [A_tag, P_tag, N_tag], # 每个都是 (3, )
            ...
        ]
        "X_dev_set": (同上)
        "X_tag_dev_set": (同上)
    }
    ```
    """
    if not os.path.exists(H5_PATH):
        print("未发现处理好的数据文件，正在处理...")
        data_set = get_data_set(amplify)
        h5f = h5py.File(H5_PATH, 'w')
        h5f["X_train_set"] = data_set["X_train_set"]
        h5f["X_tag_train_set"] = data_set["X_tag_train_set"]
        h5f["X_dev_set"] = data_set["X_dev_set"]
        h5f["X_tag_dev_set"] = data_set["X_tag_dev_set"]
        h5f.close()
    else:
        print("发现处理好的数据文件，正在读取...")
        h5f = h5py.File(H5_PATH, 'r')
        data_set = {}
        data_set["X_train_set"] = h5f["X_train_set"]
        data_set["X_tag_train_set"] = h5f["X_tag_train_set"]
        data_set["X_dev_set"] = h5f["X_dev_set"]
        data_set["X_tag_dev_set"] = h5f["X_tag_dev_set"]
        h5f.close()
    return data_set
