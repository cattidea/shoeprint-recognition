import os
import json
import h5py
import pickle

from config_parser.config import CACHE_LOADER_DIR


class CacheNotFoundError(Exception):

    def __init__(self, name):
        self.message = "Can not found cache {}".format(name)


class Cacher():
    """ Python 通用数据缓存器 """

    def __init__(self, path, type=".pkl", name=None):
        self.path = path
        self.type = type
        self.name = name

    def read(self):
        if os.path.exists(self.path):
            data = self._load()
            if self.name:
                print("Cache >> {}".format(self.name))
        else:
            raise CacheNotFoundError(self.path)
        return data

    def save(self, data):
        self._dump(data)
        if self.name:
            print("Cache << {}".format(self.name))

    def _load(self):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _dump(self, data):
        with open(self.path, 'wb') as f:
            pickle.dump(data, f, protocol = 4)


class H5Cacher(Cacher):
    """ H5 数据缓存器 """

    def __init__(self, path, name=None):
        super().__init__(path, ".h5", name)

    def _load(self):
        data = {}
        try:
            h5f = h5py.File(self.path, 'r')
            for key in h5f.keys():
                data[key] = h5f[key][:]
        finally:
            h5f.close()
        return data

    def _dump(self, data):
        try:
            h5f = h5py.File(self.path, 'w')
            for key in data.keys():
                h5f[key] = data[key]
        finally:
            h5f.close()


class JSONCacher(Cacher):
    """ JSON 数据缓存器 """

    def __init__(self, path, name=None):
        super().__init__(path, ".json", name)

    def _load(self):
        with open(self.path, 'r', encoding="utf8") as f:
            data = json.load(f)
        return data

    def _dump(self, data):
        with open(self.path, 'w', encoding="utf8") as f:
            json.dump(data, f, indent=2)


class CacheLoader():
    """ 数据装饰器
    获取数据前先检查是否有本地 Cache ，若无则重新获取并保存
    暂时只支持全数据缓存，不支持子数据缓存 """

    def __init__(self, name, debug=True, type=".pkl"):
        self.name = name
        self.debug = debug
        self.type = type

    def get_file_path(self, *args, **kw):
        file_name = self.name
        for arg in args:
            if isinstance(arg, str) and len(arg) <= 10:
                file_name += "_" + arg
        for k in kw:
            if isinstance(kw[k], str) and len(kw[k]) <= 10:
                file_name += "_" + kw[k]
        file_name += self.type
        file_path = os.path.join(CACHE_LOADER_DIR, file_name)
        return file_path

    def __call__(self, func):
        def load_data(*args, **kw):
            file_path = self.get_file_path(*args, **kw)
            cacher = Cacher(file_path, type=self.type, name=self.name)

            if self.debug and os.path.exists(file_path):
                data = cacher.read()
            else:
                data = func(*args, **kw)
                cacher.save(data)
            return data
        return load_data
