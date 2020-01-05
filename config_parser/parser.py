import json
import os


class Config(dict):
    """ 配置基类
    继承于 dict，也可直接使用属性进行读写，避免大量使用字符串的繁琐行为
    子 dict 结点也是一个 Config 结点
    """

    def __init__(self, **kw):
        super().__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                r"'Config' object has no attribute '%s'" % key)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict):
            return Config(**value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def linear(self):
        return config_to_linear(self)

    def init_path(self):
        for key in self:
            self[key] = os.path.normpath(self[key])
            if key.endswith("_dir"):
                self.touch_dir(self[key])
            else:
                self.touch_dir(os.path.dirname(self[key]))

    @staticmethod
    def touch_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)


def config_to_linear(config, linear_config={}, prefix=[]):
    for key in config:
        if isinstance(config[key], dict):
            config_to_linear(
                config[key], linear_config=linear_config, prefix=prefix+[key])
        else:
            linear_config['-'.join(prefix+[key])] = config[key]
    return linear_config


def config_to_tree(linear_config):
    config = {}
    cur = config
    for key in linear_config:
        areas = key.split('-')
        for area in areas[: -1]:
            if cur.get(area) is None:
                cur[area] = dict()
            cur = cur[area]
        cur[areas[-1]] = linear_config[key]
        cur = config
    return config


def new_config(path):
    with open(path, 'r', encoding="utf8") as f:
        json_object = json.load(f)
    return Config(**json_object)
