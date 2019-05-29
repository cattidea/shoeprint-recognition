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
    """ 获取鞋印文件结构，将鞋印图片预处理成所需格式，并将数据分类为训练类型、开发类型
    之所以不整体打乱，是因为验证集与训练集、开发集是与验证集在不同的样式中，
    所以开发集理应与训练集也在不同的样式中
    ``` json
    {
        "name": {
            "type_num": "xxxxxxxx",
            "imgs": [img1, img2, img3, ...],
            "set_type": "train/dev"
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
        set_type = "train" if random.random() < 0.95 else "dev"
        type_dir = os.path.join(SHOEPRINT_DIR, tp)
        for filename in os.listdir(type_dir):
            img_path = os.path.join(type_dir, filename)
            shoeprint_arrays[filename] = {}
            shoeprint_arrays[filename]["type_num"] = tp
            shoeprint_arrays[filename]["imgs"] = image2array(
                img_path, rotate, transpose)
            shoeprint_arrays[filename]["set_type"] = set_type
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
    """ 获取图片三元组， 可对数据进行扩增 amplify 倍 ，并且分成训练三元组和开发三元组
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
    train_img_three_tuples = []
    dev_img_three_tuples = []

    for i, img_name in enumerate(determine_scope):
        print("get_img_three_tuples {}/{}".format(i, len(determine_scope)), end='\r')
        if img_name in shoeprint_arrays:
            positive_type_num = shoeprint_arrays[img_name]["type_num"]
            set_type = shoeprint_arrays[img_name]["set_type"]
            for negative_type_num in determine_scope[img_name]:
                if negative_type_num == positive_type_num:
                    continue

                img_three_tuple_list = [(a, p, n) for a in shoeprint_arrays[img_name]["imgs"]
                                                  for p in simple_arrays[positive_type_num]["imgs"]
                                                  for n in simple_arrays[negative_type_num]["imgs"]]

                if amplify:
                    img_three_tuple_list = random.sample(
                        img_three_tuple_list, amplify)

                if set_type == "train":
                    train_img_three_tuples.extend(img_three_tuple_list)
                elif set_type == "dev":
                    dev_img_three_tuples.extend(img_three_tuple_list)
    random.shuffle(train_img_three_tuples)
    random.shuffle(dev_img_three_tuples)
    return train_img_three_tuples, dev_img_three_tuples


def get_data_set(data_set, img_three_tuples, type="train"):
    """ 将三元组数据转化为数据集格式
    ``` h5
    {
        "X": [
            [A_img, ...],
            [P_img, ...],
            [N_img, ...] # 每个都是 (78, 30, 1)
            ]
        "X_tag": [
            [A_tag, ...],
            [P_tag, ...],
            [N_tag, ...], # 每个都是 (3, )
        ]
    }
    ```
    """

    length = len(img_three_tuples)

    X = np.zeros(shape=(3, length, H, W, 1), dtype=np.bool_)
    X_tag = np.zeros(shape=(3, length, 3), dtype=np.bool_)

    for i in range(length):
        for j in range(3):
            X[j][i], X_tag[j][i] = img_three_tuples[i][j]

    data_set["X_" + type + "_set"] = X
    data_set["X_tag_" + type + "_set"] = X_tag
    return data_set

def data_import(amplify=0):
    """ 导入数据集， 分为训练集、开发集
    ``` h5
    {
        "X_train_set": [
            [A_img, ...],
            [P_img, ...],
            [N_img, ...] # 每个都是 (78, 30, 1)
            ]
        "X_tag_train_set": [
            [A_tag, ...],
            [P_tag, ...],
            [N_tag, ...], # 每个都是 (3, )
        ]
        "X_dev_set": (同上)
        "X_tag_dev_set": (同上)
    }
    ```
    """
    data_set = {}
    if not os.path.exists(H5_PATH):
        print("未发现处理好的数据文件，正在处理...")
        train_img_three_tuples, dev_img_three_tuples = get_img_three_tuples(amplify)
        data_set = get_data_set(data_set, train_img_three_tuples, type="train")
        data_set = get_data_set(data_set, dev_img_three_tuples, type="dev")
        h5f = h5py.File(H5_PATH, 'w')
        h5f["X_train_set"] = data_set["X_train_set"]
        h5f["X_tag_train_set"] = data_set["X_tag_train_set"]
        h5f["X_dev_set"] = data_set["X_dev_set"]
        h5f["X_tag_dev_set"] = data_set["X_tag_dev_set"]
        h5f.close()
    else:
        print("发现处理好的数据文件，正在读取...")
        h5f = h5py.File(H5_PATH, 'r')
        data_set["X_train_set"] = h5f["X_train_set"][: ]
        data_set["X_tag_train_set"] = h5f["X_tag_train_set"][: ]
        data_set["X_dev_set"] = h5f["X_dev_set"][: ]
        data_set["X_tag_dev_set"] = h5f["X_tag_dev_set"][: ]
        h5f.close()
    train_length = len(data_set["X_train_set"][0])
    dev_length = len(data_set["X_dev_set"][0])
    print("成功加载训练集 {} 条，开发集 {} 条".format(train_length, dev_length))
    return data_set
