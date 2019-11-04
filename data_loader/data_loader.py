import os
import json
import h5py
import random
import numpy as np

from config_parser.config import PATHS, DEBUG
from data_loader.base import CacheLoader
from data_loader.image import image2array

H5_PATH = PATHS["h5_path"]
JSON_PATH = PATHS["json_path"]
SHOEPRINT_DIR = PATHS["shoeprint_dir"]
SAMPLE_DIR = PATHS["sample_dir"]
SHOEPRINT_DIR_TEST = PATHS["shoeprint_test_dir"]
DETERMINE_FILE = PATHS["determine_file"]
DETERMINE_FILE_TEST = PATHS["determine_test_file"]


@CacheLoader(name="sample", debug=DEBUG)
def get_sample_arrays(amplify):
    """ 获取样本文件结构，将样本图片预处理成所需格式
    ``` python
    [
        [<img1_array1>, <img1_array2>, ...],
        [<img2_array1>, <img2_array2>, ...],
        ...
    ],
    {
        <type_id>: {
            "img_indices": [<img1_index>, <img2_index>, <img3_index>, ...],
        },
        ...
    },
    ```
    """
    sample_map = {}
    sample_arrays = []
    types = os.listdir(SAMPLE_DIR)
    index = 0
    assert types, "样本图库文件夹为空！"
    for i, type_id in enumerate(types):
        print("get_sample_arrays {}/{} ".format(i, len(types)), end='\r')
        type_dir = os.path.join(SAMPLE_DIR, type_id)
        img_path = os.path.join(type_dir, os.listdir(type_dir)[0])
        sample_map[type_id] = {}

        img_array = image2array(img_path, amplify)

        sample_map[type_id]["img_indices"] = [index + j for j in range(len(img_array))]
        index += len(img_array)
        sample_arrays.extend(img_array)
    assert len(sample_arrays) == index
    return sample_arrays, sample_map


@CacheLoader(name="shoeprint", debug=DEBUG)
def get_shoeprint_arrays(amplify, sample_length, action_type="train"):
    """ 获取鞋印文件结构，将鞋印图片预处理成所需格式追加在 sample_arrays 后，并将数据分类为训练类型、开发类型
    之所以不整体打乱，是因为验证集与训练集、开发集是与验证集在不同的样式中，
    所以开发集理应与训练集也在不同的样式中
    ``` python
    [
        [<img1_array1>, <img1_array2>, ...],
        [<img2_array1>, <img2_array2>, ...],
        ...
    ],
    {
        <name>: {
            "type_id": <xxxxxxxx>,
            "img_indices": [<img1_index>, <img2_index>, <img3_index>, ...],
            "set_type": "train/dev/test"
        },
        ...
    }
    {
        <type_id1>: [<name1>, <name2>, ...],
        <type_id2>: [<name1>, <name2>, ...],
        ...
    }
    ```
    """
    shoeprint_map = {}
    shoeprint_arrays = []
    type_map = {}
    shoeprint_base_dir = SHOEPRINT_DIR if action_type == "train" else SHOEPRINT_DIR_TEST
    types = os.listdir(shoeprint_base_dir)
    type_counter = {"train": set(), "dev": set(), "test": set()}
    index = sample_length
    assert types, "鞋印图库文件夹为空！"

    for i, type_id in enumerate(types):
        print("get_shoeprint_arrays {}/{} ".format(i, len(types)), end='\r')
        if action_type == "train":
            set_type = "train" if random.random() < 0.95 else "dev"
        else:
            set_type = "test"
        type_dir = os.path.join(shoeprint_base_dir, type_id)
        type_map[type_id] = []
        for filename in os.listdir(type_dir):
            img_path = os.path.join(type_dir, filename)
            img_array = image2array(img_path, amplify)
            shoeprint_map[filename] = {}
            shoeprint_map[filename]["type_id"] = type_id
            shoeprint_map[filename]["img_indices"] = [index + j for j in range(len(img_array))]
            shoeprint_map[filename]["set_type"] = set_type
            shoeprint_arrays.extend(img_array)
            index += len(img_array)

            type_counter[set_type].add(type_id)
            type_map[type_id].append(filename)
    if action_type == "train":
        print("训练数据共 {} 类，开发数据共 {} 类".format(len(type_counter["train"]), len(type_counter["dev"])))
    else:
        print("测试数据共 {} 类".format(len(type_counter["test"])))
    assert len(shoeprint_arrays) == index - sample_length
    return shoeprint_arrays, shoeprint_map, type_map


@CacheLoader(name="determine", debug=DEBUG)
def get_determine_scope(action_type="train"):
    """ 读取待判定范围文件，并构造成字典型
    ``` python
    {
        <name>: [
            <P>, <N1>, <N2>, <N3>, ... // 注意， P 不一定在最前面，而且这里记录的是 type_id
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


@CacheLoader(name="class_indices", debug=DEBUG)
def get_indices(sample_map, shoeprint_map, type_map):
    """ 将所有 indices 组织在一起
    ``` python
    [
        [
            [<idx01>, <idx02>], # 某一个
            [<idx01>, <idx02>],
            ...
        ], # 某一类
        ...
    ]
    ```
    """
    indices = []
    for i, type_id in enumerate(sample_map):
        print("get_indices {}/{} ".format(i, len(sample_map)), end='\r')
        class_indices = []
        class_indices.append(sample_map[type_id]["img_indices"])
        if type_id in type_map:
            for pos_name in type_map[type_id]:
                if shoeprint_map[pos_name]["set_type"] == "train":
                    class_indices.append(shoeprint_map[pos_name]["img_indices"])
        indices.append(class_indices)
    return indices


@CacheLoader(name="test_data_set", debug=DEBUG)
def test_data_import(amplify=[], action_type="test"):
    """ 构造测试数据
    ``` python
    img_arrays
    {
        "train": [
            {
                "name": <name>,
                "index": <idx>,
                "scope_indices": [<idx01>, <idx02>, ...],
                "label": <correct_idx>
            },
            ...
        ],
        "dev": ...,
        "test": ...
    }
    ```
    """

    determine_scope = get_determine_scope(action_type=action_type)
    sample_arrays, sample_map = get_sample_arrays(amplify=[])
    shoeprint_arrays, shoeprint_map, _ = get_shoeprint_arrays(
        amplify=amplify, sample_length=len(sample_arrays), action_type=action_type)
    img_arrays = np.concatenate((sample_arrays, shoeprint_arrays))
    test_data_map = {"train": [], "dev": [], "test": []}

    print("sample {} shoeprint {} ".format(len(sample_arrays), len(shoeprint_arrays)))

    scope_length = len(determine_scope[list(determine_scope.keys())[0]])
    imgs_num = len(determine_scope)

    for i, origin_name in enumerate(determine_scope):
        print("get_test_data ({}) {}/{} ".format(action_type, i, imgs_num), end='\r')
        if action_type == "test":
            assert origin_name in shoeprint_map
        else:
            if origin_name not in shoeprint_map:
                print(origin_name)
                continue

        set_type = shoeprint_map[origin_name]["set_type"]
        type_id = shoeprint_map[origin_name]["type_id"]
        item = {}
        item["name"] = origin_name
        item["indices"] = shoeprint_map[origin_name]["img_indices"]
        item["scope_indices"] = []
        item["label"] = determine_scope[origin_name].index(type_id)
        for j in range(scope_length):
            item["scope_indices"].append(sample_map[determine_scope[origin_name][j]]["img_indices"][0])
        test_data_map[set_type].append(item)
    return img_arrays, test_data_map, len(sample_arrays)


def data_import(amplify=[]):
    """ 导入数据集， 分为训练集、开发集
    ``` h5
    {
        "img_arrays": [<img01>, <img02>, ...] # 每个都是 (H, W, 1)
    }
    ```
    """
    data_set = {}
    if not os.path.exists(H5_PATH) or not os.path.exists(JSON_PATH):
        print("未发现处理好的数据文件，正在处理...")
        determine_scope = get_determine_scope(action_type="train")
        sample_arrays, sample_map = get_sample_arrays(amplify)
        shoeprint_arrays, shoeprint_map, type_map = get_shoeprint_arrays(
            amplify, sample_length=len(sample_arrays), action_type="train")
        img_arrays = np.concatenate((sample_arrays, shoeprint_arrays))
        indices = get_indices(sample_map, shoeprint_map, type_map)

        data_set["img_arrays"] = img_arrays
        data_set["indices"] = indices

        h5f = h5py.File(H5_PATH, 'w')
        h5f["img_arrays"] = data_set["img_arrays"]
        h5f.close()

        with open(JSON_PATH, 'w', encoding="utf8") as f:
            json.dump(data_set["indices"], f, indent=2)
    else:
        print("发现处理好的数据文件，正在读取...")
        h5f = h5py.File(H5_PATH, 'r')
        data_set["img_arrays"] = h5f["img_arrays"][: ]
        h5f.close()

        with open(JSON_PATH, 'r', encoding="utf8") as f:
            data_set["indices"] = json.load(f)
    print("成功加载数据")
    return data_set
