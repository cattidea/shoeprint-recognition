import os
import random
import pickle
import itertools

import h5py
import numpy as np

from utils.config import Config
from utils.imager import H, W, image2array, image2array_v2

CONFIG = Config()
SIMPLE_DIR = CONFIG['simple_dir']
DETERMINE_FILE = CONFIG["determine_file"]
DETERMINE_FILE_TEST = CONFIG["determine_file_test"]
SHOEPRINT_DIR = CONFIG["shoeprint_dir"]
SHOEPRINT_DIR_TEST = CONFIG["shoeprint_dir_test"]
H5_PATH = CONFIG["h5_path"]
DEBUG_FILE = CONFIG['debug_file']
SEED = 1111
random.seed(SEED)


def get_simple_arrays(amplify):
    """ 获取样本文件结构，将样本图片预处理成所需格式
    ``` python
    simple_arrays
    {
        "type_id": {
            "img_indices": [img1_index, img2_index, img3_index, ...],
        },
        ...
    },
    ```
    """
    rotate, transpose = bool(amplify), bool(amplify)
    simple_map = {}
    simple_arrays = []
    types = os.listdir(SIMPLE_DIR)
    index = 0
    for i, type_id in enumerate(types):
        print("get_simple_arrays {}/{} ".format(i, len(types)), end='\r')
        type_dir = os.path.join(SIMPLE_DIR, type_id)
        img_path = os.path.join(type_dir, os.listdir(type_dir)[0])
        simple_map[type_id] = {}

        img_array = image2array(img_path, rotate, transpose)

        simple_map[type_id]["img_indices"] = [index + j for j in range(len(img_array))]
        index += len(img_array)
        simple_arrays.extend(img_array)
    assert len(simple_arrays) == index
    return simple_arrays, simple_map


def get_shoeprint_arrays(simple_arrays, amplify, action_type="train"):
    """ 获取鞋印文件结，将鞋印图片预处理成所需格式追加在 simple_arrays 后，并将数据分类为训练类型、开发类型
    之所以不整体打乱，是因为验证集与训练集、开发集是与验证集在不同的样式中，
    所以开发集理应与训练集也在不同的样式中
    ``` python
    shoeprint_arrays
    {
        "name": {
            "type_id": "xxxxxxxx",
            "img_indices": [img1_index, img2_index, img3_index, ...],
            "set_type": "train/dev/test"
        },
        ...
    }
    {
        "type_id1": ["name1", "name2", ...],
        "type_id2": ["name1", "name2", ...],
        ...
    }
    ```
    """
    rotate, transpose = bool(amplify), bool(amplify)
    shoeprint_start = len(simple_arrays)
    shoeprint_map = {}
    shoeprint_arrays = []
    type_map = {}
    types = os.listdir(SHOEPRINT_DIR) if action_type == "train" else os.listdir(SHOEPRINT_DIR_TEST)
    type_counter = {"train": set(), "dev": set(), "test": set()}
    index = 0

    for i, type_id in enumerate(types):
        print("get_shoeprint_arrays {}/{} ".format(i, len(types)), end='\r')
        if action_type == "train":
            set_type = "train" if random.random() < 0.95 else "dev"
        else:
            set_type = "test"
        type_dir = os.path.join(SHOEPRINT_DIR, type_id)
        type_map[type_id] = []
        for filename in os.listdir(type_dir):
            img_path = os.path.join(type_dir, filename)
            img_array = image2array(img_path, rotate, transpose)
            shoeprint_map[filename] = {}
            shoeprint_map[filename]["type_id"] = type_id
            shoeprint_map[filename]["img_indices"] = [index + j + shoeprint_start for j in range(len(img_array))]
            shoeprint_map[filename]["set_type"] = set_type
            shoeprint_arrays.extend(img_array)
            index += len(img_array)

            type_counter[set_type].add(type_id)
            type_map[type_id].append(filename)
    if action_type == "train":
        print("训练数据共 {} 类，开发数据共 {} 类".format(len(type_counter["train"]), len(type_counter["dev"])))
    else:
        print("测试数据共 {} 类".format(len(type_counter["test"])))
    assert len(shoeprint_arrays) == index
    return np.concatenate((simple_arrays, shoeprint_arrays)), shoeprint_map, type_map


def get_determine_scope(action_type="train"):
    """ 读取待判定范围文件，并构造成字典型
    ``` python
    {
        "name": [
            P, N1, N2, N3, ... // 注意， P 不一定在最前面，而且这里记录的是 type_id
        ],
        ...
    }
    ```
    """
    determine_scope = {}
    determine_scope_file = DETERMINE_FILE if action_type == "train" else DETERMINE_FILE_TEST
    with open(determine_scope_file, 'r') as f:
        for line in f:
            line_items = line.split('\t')
            for i in range(len(line_items)):
                line_items[i] = line_items[i].strip()
            determine_scope[line_items[0]] = line_items[1:]
    return determine_scope


def get_img_triplets(amplify):
    """ 获取图片三元组， 可对数据进行扩增 amplify 倍 ，并且分成训练三元组和开发三元组
    ``` python
    [
        (
            A_idx,
            P_idx,
            N_idx
        ),
        ...
    ]
    ```
    """

    determine_scope = get_determine_scope(action_type="train")
    simple_arrays, simple_map = get_simple_arrays(amplify)
    img_arrays, shoeprint_map, type_map = get_shoeprint_arrays(simple_arrays, amplify, action_type="train")
    train_img_triplets = []
    dev_img_triplets = []

    for i, type_id in enumerate(type_map):
        print("get_img_triplets {}/{} ".format(i, len(type_map)), end='\r')
        img_names = type_map[type_id]
        set_type = shoeprint_map[img_names[0]]["set_type"]
        negative_type_ids = []

        for img_name in img_names:
            negative_type_ids_block = list(determine_scope[img_name])
            if img_name in negative_type_ids_block:
                negative_type_ids_block.pop(negative_type_ids_block.index(img_name))
            negative_type_ids.extend(negative_type_ids_block)
        negative_type_ids = list(set(negative_type_ids))

        positive_indices = []
        negative_indices = []

        positive_indices.extend(simple_map[type_id]["img_indices"] * 5)
        for img_name in img_names:
            positive_indices.extend(shoeprint_map[img_name]["img_indices"])

        for negative_type_id in negative_type_ids:
            negative_indices.extend(simple_map[negative_type_id]["img_indices"] * 5)
            for negative_name in type_map.get(type_id, []):
                negative_indices.extend(shoeprint_map[negative_name]["img_indices"])

        img_triplets_block = [
            (*a_p, n)
            for a_p in itertools.combinations(positive_indices, 2)
            for n in negative_indices
        ]

        if amplify:
            img_triplets_block = random.sample(img_triplets_block, 1000 * amplify)
        else:
            img_triplets_block = random.sample(img_triplets_block, min(len(img_triplets_block), 1000))

        if set_type == "train":
            train_img_triplets.extend(img_triplets_block)
        elif set_type == "dev":
            dev_img_triplets.extend(img_triplets_block)

    random.shuffle(train_img_triplets)
    random.shuffle(dev_img_triplets)
    return train_img_triplets, dev_img_triplets, img_arrays


def get_test_data_set():
    """ 构造测试集数据
    ``` python
    img_arrays
    {
        "name": {
            "index": idx,
            "scope_indices": [idx01, idx02, ...],
            "label": correct_idx
        }
    }
    ```
    """

    amplify = 0
    determine_scope = get_determine_scope(action_type="test")
    simple_arrays, simple_map = get_simple_arrays(amplify)
    img_arrays, shoeprint_map, _ = get_shoeprint_arrays(simple_arrays, amplify, action_type="test")
    test_data_map = {}

    scope_length = len(determine_scope[list(determine_scope.keys())[0]])
    imgs_num = len(determine_scope)

    for i, origin_name in enumerate(determine_scope):
        print("get_test_data_set {}/{} ".format(i, imgs_num), end='\r')
        assert origin_name in shoeprint_map
        type_id = shoeprint_map[origin_name]["type_id"]

        test_data_map[origin_name] = {}
        test_data_map[origin_name]["index"] = shoeprint_map[origin_name]["img_indices"][0]
        test_data_map[origin_name]["scope_indices"] = []
        test_data_map[origin_name]["label"] = determine_scope[origin_name].index(type_id)
        for j in range(scope_length):
            test_data_map[origin_name]["scope_indices"].append(simple_map[determine_scope[origin_name][j]]["img_indices"][0])
    return img_arrays, test_data_map


def get_data_set(data_set, img_triplets, type="train"):
    """ 将三元组数据转化为数据集格式
    ``` h5
    {
        "X_indices": [
            [A_idx, ...],
            [P_idx, ...],
            [N_idx, ...]
            ]
        ]
    }
    ```
    """

    length = len(img_triplets)

    X = np.zeros(shape=(3, length), dtype=np.int32)

    for i in range(length):
        for j in range(3):
            print("get_data_set {}/{} ".format(i * 3 + j, length * 3), end='\r')
            X[j][i] = img_triplets[i][j]

    data_set["X_indices_" + type + "_set"] = X
    return data_set


def data_import(amplify=0):
    """ 导入数据集， 分为训练集、开发集
    ``` h5
    {
        "X_indices_train_set": [
            [A_idx, ...],
            [P_idx, ...],
            [N_idx, ...] # 每个都是 (H, W, 1)
            ]
        "X_indices_dev_set": (同上),
        "X_imgs": [img01, img02, ...] # 每个都是 (H, W, 1)
    }
    ```
    """
    data_set = {}
    if not os.path.exists(H5_PATH):
        print("未发现处理好的数据文件，正在处理...")
        train_img_triplets, dev_img_triplets, data_set["X_imgs"] = get_img_triplets(amplify)
        data_set = get_data_set(data_set, train_img_triplets, type="train")
        data_set = get_data_set(data_set, dev_img_triplets, type="dev")
        h5f = h5py.File(H5_PATH, 'w')
        h5f["X_indices_train_set"] = data_set["X_indices_train_set"]
        h5f["X_indices_dev_set"] = data_set["X_indices_dev_set"]
        h5f["X_imgs"] = data_set["X_imgs"]
        h5f.close()
    else:
        print("发现处理好的数据文件，正在读取...")
        h5f = h5py.File(H5_PATH, 'r')
        data_set["X_indices_train_set"] = h5f["X_indices_train_set"][: ]
        data_set["X_indices_dev_set"] = h5f["X_indices_dev_set"][: ]
        data_set["X_imgs"] = h5f["X_imgs"][: ]
        h5f.close()
    train_length = len(data_set["X_indices_train_set"][0])
    dev_length = len(data_set["X_indices_dev_set"][0])
    print("成功加载训练集 {} 条，开发集 {} 条".format(train_length, dev_length))
    return data_set

def test_data_import(debug=False):
    if debug and os.path.exists(DEBUG_FILE):
        with open(DEBUG_FILE, 'rb') as f:
            img_arrays, test_data_map = pickle.load(f)
    else:
        img_arrays, test_data_map = get_test_data_set()
        if debug:
            with open(DEBUG_FILE, 'wb') as f:
                pickle.dump((img_arrays, test_data_map), f)
    return img_arrays, test_data_map
