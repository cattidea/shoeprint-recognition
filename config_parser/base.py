import json
import os


class Config(dict):
    """ 配置基类 """

    def __init__(self, path=None, config={}):
        if path is not None and os.path.exists(path):
            config.update(self.read(path))
        super().__init__(config)


    def read(self, path):
        with open(path, 'r', encoding="utf8") as f:
            _jObject = json.load(f)
        return _jObject


    def sub_config(self, name, cls=None):
        if cls is None:
            cls = type(self)
        return cls(config=self[name])


class PathConfig(Config):
    """ 路径配置项
    可以在初始化时确保所有需要的文件夹（以"_dir"结尾的键）已经建立 """

    def __init__(self, path=None, config={}):
        super().__init__(path, config)
        for name in self:
            self[name] = os.path.normpath(self[name])
            if name.endswith("_dir"):
                self.touch_dir(self[name])
            else:
                self.touch_dir(os.path.dirname(self[name]))


    @staticmethod
    def touch_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
