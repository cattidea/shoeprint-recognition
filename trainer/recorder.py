class Recorder():
    """ 训练记录器 """
    def __init__(self, path, resume=False, separator=","):
        self.path = path
        self.separator = separator
        if resume:
            self._parse()
        else:
            self._init()

    def _parse(self):
        """ 从 csv 中解析数据 """
        self.checkpoint = 0
        self.params = {}
        self.record = []

        csv_list = []
        with open(self.path, "r", encoding="utf8") as f:
            for line in f:
                csv_list.append(line.rstrip("\n").split(self.separator))

        tp = 0
        for csv_line in csv_list:
            if csv_line[0] == "":
                tp += 1
                continue
            if tp == 0:
                self.checkpoint = int(csv_line[0])
            elif tp == 1:
                self.params[csv_line[0]] = {}
                for item in csv_line[1: ]:
                    if item == "":
                        continue
                    value = item.split("[")[0]
                    index = int(item.split("[")[1].rstrip("]"))
                    self.params[csv_line[0]][index] = value
            elif tp == 2:
                self.record.append(csv_line)

    def save(self):
        """ 存储 """
        csv_list = []

        csv_list.append([str(self.checkpoint)])

        csv_list.append([])

        for key, param in self.params.items():
            param_line_list = []
            for index, value in param.items():
                param_line_list.append("{}[{}]".format(value, index))
            csv_list.append([key, *param_line_list])

        csv_list.append([])

        for record_line_list in self.record:
            record_line_list = list(map(lambda x: str(x), record_line_list))
            csv_list.append(record_line_list)

        csv_list.append([])

        with open(self.path, "w", encoding="utf8") as f:
            for csv_line_list in csv_list:
                f.write(self.separator.join(csv_line_list) + "\n")

    def _init(self):
        """ 初始化空参数 """
        self.checkpoint = 0
        self.params = {}
        self.record = []

    def update_checkpoint(self, checkpoint):
        """ 更新断点 """
        self.checkpoint = checkpoint

    def upload_params(self, params):
        """ 更新参数 """
        for key, value in params.items():
            if self.params.get(key, None) is None:
                self.params[key] = {}
            self.params[key][self.checkpoint] = value

    def record_item(self, checkpoint, item):
        """ 记录一项数据 """
        self.checkpoint = checkpoint
        self.record.append([checkpoint, *item])
