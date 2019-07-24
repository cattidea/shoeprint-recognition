import itertools
import json
import os
import pickle
import random

import h5py
import numpy as np

from utils.config import Config
from utils.imager import H, W, image2array, TRANSPOSE
from utils.nn import compute_embeddings

CONFIG = Config()
SIMPLE_DIR = CONFIG['simple_dir']
DETERMINE_FILE = CONFIG["determine_file"]
DETERMINE_FILE_TEST = CONFIG["determine_file_test"]
SHOEPRINT_DIR = CONFIG["shoeprint_dir"]
SHOEPRINT_DIR_TEST = CONFIG["shoeprint_dir_test"]
H5_PATH = CONFIG["h5_path"]
JSON_PATH = CONFIG["json_path"]
DATA_LOADER_DIR = CONFIG["data_loader_dir"]
SEED = 1111
random.seed(SEED)
np.random.seed(SEED)


def data_loader(name, debug=True):
    """ 数据装饰器，获取数据前先检查是否有本地 Cache ，若无则重新获取并保存 """
    def load_data(func):
        def inner_load_data(*args, **kw):
            file_name = name
            for arg in args:
                if isinstance(arg, str) and len(arg) <= 10:
                    file_name += "_" + arg
            for k in kw:
                if isinstance(kw[k], str) and len(kw[k]) <= 10:
                    file_name += "_" + kw[k]
            file_name += ".dat"
            file_path = os.path.join(DATA_LOADER_DIR, file_name)
            if debug and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = func(*args, **kw)
                if debug:
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
            return data
        return inner_load_data
    return load_data


@data_loader(name="simple", debug=True)
def get_simple_arrays(amplify):
    """ 获取样本文件结构，将样本图片预处理成所需格式
    ``` python
    [
        [<img1_array1>, <img1_array2>, ...],
        [<img2_array1>, <img2_array2>, ...],
        ...
    ],
    [
        [<img1_mask1>, <img1_mask2>, ...],
        [<img2_mask1>, <img2_mask2>, ...],
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
    simple_map = {}
    simple_arrays = []
    simple_masks = []
    types = os.listdir(SIMPLE_DIR)
    index = 0
    for i, type_id in enumerate(types):
        print("get_simple_arrays {}/{} ".format(i, len(types)), end='\r')
        type_dir = os.path.join(SIMPLE_DIR, type_id)
        img_path = os.path.join(type_dir, os.listdir(type_dir)[0])
        simple_map[type_id] = {}

        img_array, masks = image2array(img_path, amplify)

        simple_map[type_id]["img_indices"] = [index + j for j in range(len(img_array))]
        index += len(img_array)
        simple_arrays.extend(img_array)
        simple_masks.extend(masks)
    assert len(simple_arrays) == index
    return simple_arrays, simple_masks, simple_map


@data_loader(name="shoeprint", debug=True)
def get_shoeprint_arrays(amplify, simple_length, action_type="train"):
    """ 获取鞋印文件结，将鞋印图片预处理成所需格式追加在 simple_arrays 后，并将数据分类为训练类型、开发类型
    之所以不整体打乱，是因为验证集与训练集、开发集是与验证集在不同的样式中，
    所以开发集理应与训练集也在不同的样式中
    ``` python
    [
        [<img1_array1>, <img1_array2>, ...],
        [<img2_array1>, <img2_array2>, ...],
        ...
    ],
    [
        [<img1_mask1>, <img1_mask2>, ...],
        [<img2_mask1>, <img2_mask2>, ...],
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
    shoeprint_masks = []
    type_map = {}
    types = os.listdir(SHOEPRINT_DIR) if action_type == "train" else os.listdir(SHOEPRINT_DIR_TEST)
    type_counter = {"train": set(), "dev": set(), "test": set()}
    index = simple_length

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
            img_array, masks = image2array(img_path, amplify)
            shoeprint_map[filename] = {}
            shoeprint_map[filename]["type_id"] = type_id
            shoeprint_map[filename]["img_indices"] = [index + j for j in range(len(img_array))]
            shoeprint_map[filename]["set_type"] = set_type
            shoeprint_arrays.extend(img_array)
            shoeprint_masks.extend(masks)
            index += len(img_array)

            type_counter[set_type].add(type_id)
            type_map[type_id].append(filename)
    if action_type == "train":
        print("训练数据共 {} 类，开发数据共 {} 类".format(len(type_counter["train"]), len(type_counter["dev"])))
    else:
        print("测试数据共 {} 类".format(len(type_counter["test"])))
    assert len(shoeprint_arrays) == index - simple_length
    return shoeprint_arrays, shoeprint_masks, shoeprint_map, type_map


@data_loader(name="determine", debug=True)
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


@data_loader(name="class_indices", debug=True)
def get_indices(simple_map, shoeprint_map, type_map):
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
    for i, type_id in enumerate(simple_map):
        print("get_indices {}/{} ".format(i, len(simple_map)), end='\r')
        class_indices = []
        class_indices.append(simple_map[type_id]["img_indices"])
        if type_id in type_map:
            for pos_name in type_map[type_id]:
                if shoeprint_map[pos_name]["set_type"] == "train":
                    class_indices.append(shoeprint_map[pos_name]["img_indices"])
        indices.append(class_indices)
    return indices


@data_loader(name="test_data_set", debug=True)
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
    simple_arrays, simple_masks, simple_map = get_simple_arrays(amplify=[])
    shoeprint_arrays, shoeprint_masks, shoeprint_map, _ = get_shoeprint_arrays(
        amplify=amplify, simple_length=len(simple_arrays), action_type=action_type)
    img_arrays = np.concatenate((simple_arrays, shoeprint_arrays))
    masks = np.concatenate((simple_masks, shoeprint_masks))
    test_data_map = {"train": [], "dev": [], "test": []}

    print("simple {} shoeprint {} ".format(len(simple_arrays), len(shoeprint_arrays)))

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
            item["scope_indices"].append(simple_map[determine_scope[origin_name][j]]["img_indices"][0])
        test_data_map[set_type].append(item)
    return img_arrays, masks, test_data_map


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
        simple_arrays, simple_masks, simple_map = get_simple_arrays(amplify)
        shoeprint_arrays, shoeprint_masks, shoeprint_map, type_map = get_shoeprint_arrays(
            amplify, simple_length=len(simple_arrays), action_type="train")
        img_arrays = np.concatenate((simple_arrays, shoeprint_arrays))
        masks = np.concatenate((simple_masks, shoeprint_masks))
        indices = get_indices(simple_map, shoeprint_map, type_map)

        data_set["img_arrays"] = img_arrays
        data_set["masks"] = masks
        data_set["indices"] = indices

        h5f = h5py.File(H5_PATH, 'w')
        h5f["img_arrays"] = data_set["img_arrays"]
        h5f["masks"] = data_set["masks"]
        h5f.close()

        with open(JSON_PATH, 'w', encoding="utf8") as f:
            json.dump(data_set["indices"], f, indent=2)
    else:
        print("发现处理好的数据文件，正在读取...")
        h5f = h5py.File(H5_PATH, 'r')
        data_set["img_arrays"] = h5f["img_arrays"][: ]
        data_set["masks"] = h5f["masks"][: ]
        h5f.close()

        with open(JSON_PATH, 'r', encoding="utf8") as f:
            data_set["indices"] = json.load(f)
    print("成功加载数据")
    return data_set


def sample_shoeprint(data_set, start_index, class_per_batch, shoe_per_class, img_per_shoe):
    """ 抽取一个 batch 所需的鞋印
    ``` python
    [
        <idx01>, <idx02>, ...
    ]
    ```
    """
    nrof_shoes = class_per_batch * shoe_per_class
    nrof_classes = len(data_set)
    img_per_shoe_origin = len(data_set[0][0])

    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    shoeprints = []
    nrof_shoes_per_class = []

    while len(shoeprints) < nrof_shoes:
        # print("sample_shoeprint {}/{} ".format(len(shoeprints), nrof_shoes), end='\r')
        class_index = class_indices[start_index]
        # 某一类中鞋印的总数量
        nrof_shoes_in_class = len(data_set[class_index])
        # if nrof_shoes_in_class > 1:
        if True:
            shoe_indices = np.arange(nrof_shoes_in_class)
            np.random.shuffle(shoe_indices)
            # 该类中需要抽取鞋印的数量
            nrof_shoes_from_class = min(nrof_shoes_in_class, shoe_per_class, nrof_shoes-len(shoeprints))
            idx = shoe_indices[: nrof_shoes_from_class]
            # 随机选取一定量的扩增图
            img_indices = np.random.choice(img_per_shoe_origin, img_per_shoe, replace=False)
            shoeprints += [np.array(data_set[class_index][i])[img_indices] for i in idx]
            nrof_shoes_per_class.append(nrof_shoes_from_class)

        start_index += 1
        start_index %= nrof_classes

    assert len(shoeprints) == nrof_shoes
    return np.reshape(shoeprints, (nrof_shoes * img_per_shoe, )), nrof_shoes_per_class, start_index


def select_triplets(embeddings, shoeprints, nrof_shoes_per_class, class_per_batch, img_per_shoe, alpha):
    """ 选择三元组 """
    emb_start_idx = 0
    triplets = []

    for i in range(len(nrof_shoes_per_class)):
        # print("select_triplets {}/{} ".format(i, class_per_batch), end='\r')
        nrof_shoes = int(nrof_shoes_per_class[i])
        if nrof_shoes <= 1:
            continue

        # 某个鞋
        for j in range(0, nrof_shoes*img_per_shoe, img_per_shoe):
            a_offset = np.random.randint(img_per_shoe) # 同图偏移
            a_idx = emb_start_idx + j + a_offset
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), axis=1)
            # 将本类鞋距离设为无穷，不作 negative
            neg_dists_sqr[emb_start_idx: emb_start_idx+nrof_shoes*img_per_shoe] = np.NaN

            for k in range(j+img_per_shoe, nrof_shoes*img_per_shoe, img_per_shoe):
                p_offset = np.random.randint(img_per_shoe)
                p_idx = emb_start_idx + k + p_offset
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                # 由于 neg_dist 中有 NaN ，故会有 RuntimeWarning
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]

                if nrof_random_negs > 0:
                    # 如果存在满足条件的 neg ，则随机挑选一个
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((shoeprints[a_idx], shoeprints[p_idx], shoeprints[n_idx]))

        emb_start_idx += nrof_shoes * img_per_shoe

    np.random.shuffle(triplets)
    return triplets


def gen_mini_batch(indices, class_per_batch, shoe_per_class, img_per_shoe,
                   img_arrays, masks, sess, ops, alpha=0.2, step=512):
    """ 生成 mini-batch """
    start_index = 0
    shadow_index = 0
    while start_index >= shadow_index:
        shadow_index = start_index
        shoeprints, nrof_shoes_per_class, start_index = sample_shoeprint(indices, start_index, class_per_batch, shoe_per_class, img_per_shoe)
        embeddings = compute_embeddings(img_arrays[shoeprints], masks[shoeprints], sess, ops, step=step)
        triplets = select_triplets(embeddings, shoeprints, nrof_shoes_per_class, class_per_batch, img_per_shoe, alpha)
        yield shadow_index, triplets
