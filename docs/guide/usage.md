# Usage

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

或者逐一进行安装，`tensorflow` 与 `tensorflow-gpu` 存一即可，版本号为 `1.13.1+`

### 文件配置

可按照 `config.json` 对文件进行配置，默认配置如下

```
.
├── data
│   ├── cache
│   │   ├── cache_loader/
│   │   ├── data_set.h5
│   │   └── indices.json
│   ├── ckpt
│   ├── simple_pics              （训练、测试必配置）
│   │   ├── C151224000005
│   │   ├── C151224000012
│   │   └── ...
│   ├── test
│   │   ├── test_2               （测试必配置）
│   │   │   ├── C151224000012
│   │   │   └── ...
│   │   ├── test_剪切图
│   │   └── test_原图
│   ├── train
│   │   ├── train_2              （训练必配置）
│   │   │   ├── C151224000012
│   │   │   └── ...
│   │   ├── train_剪切图
│   │   └── train_原图
│   ├── txt
│   │   ├── 索引.txt             （测试必配置）
│   │   └── 训练.txt             （训练必配置）
│   └── result.txt               （测试必配置）
├── docs/
├── scripts/
├── config_parser/
├── data_loader/
├── infer/
├── model/
├── trainer/
├── config.json
└── main.py
```

下面对各个文件进行说明

-  `docs/` 文档文件夹，使用 Markdown 书写， Vuepress 生成
-  `scripts/` 部分脚本，仅起到某些命令的简化作用，对本项目无实质性的帮助
-  `config.json` 配置文件，用于配置文件路径、训练超参数、图片参数等
-  `main.py` 入口函数，用于解析参数
-  `config_parser/` 配置解析库
   -  `base.py` 配置基类
   -  `config.py` 对 `config.json` 进行初步解析
-  `data_loader/` 数据提取库，包含以下三个主模块
   -  `base.py` 缓存装饰器，用于生成数据一级 pickle 缓存，以提高数据获取速度
   -  `data_loader.py` 原数据提取器，用于从原数据文件中提取需要的数据，大多数使用了缓存装饰器，其中 `data_import` 使用了二级缓存（h5 和 json）提高读取速度
   -  `image.py` 图片处理模块，主要用于图片的 resize 与扩增
   -  `batch_loader/` 三元组在线生成器（确实是个生成器）
-  `model/` 模型库
   -  `base.py` 模型基类
   -  `triplet_model.py` 对本问题的衍生类，封装了本问题所需要的大多数方法以及初始化方法
   -  `models.py` 模型类文件，每个类对应一个模型
-  `trainer/` 训练库，暂未进行模块化封装
   -  `train.py` 训练时的主函数文件
-  `infer/` 测试库，暂未进行模块化封装
   -  `test.py` 测试时的主函数文件
-  `data/cache/cache_loader/` `data_loader/base.py` 默认生成的缓存文件夹，可通过修改 `paths.cache_loader_dir` 进行变更
-  `data/cache/data_set.h5` `data_loader/data_loader.data_import` 默认生成的二级图片缓存，可通过修改 `paths.h5_path` 进行变更
-  `data/cache/indices.json` `data_loader/data_loader.data_import` 默认生成的二级索引缓存，可通过修改 `paths.json_path` 进行变更
-  `data/ckpt/` 模型文件夹，训练时会将模型保存在该文件夹下，测试时会使用该文件夹下的模型，可通过修改 `paths.model_dir` 进行变更
-  `data/simple_pics/` 样本图库文件夹，请训练与测试时将图片存放到该文件夹下，可通过修改 `paths.simple_dir` 进行变更
-  `data/train/train_2/` 训练鞋印图库文件夹，可通过修改 `paths.shoeprint_dir` 进行变更
-  `data/test/test_2/` 测试鞋印图库文件夹，可通过修改 `paths.shoeprint_test_dir` 进行变更
-  `data/txt/训练.txt` 训练待判定范围文本文件，可通过修改 `paths.determine_file` 进行变更
-  `data/txt/索引.txt` 测试待判定范围文本文件，可通过修改 `paths.determine_test_file` 进行变更
-  `result.txt` 测试输出文本文件，可通过修改 `paths.result_file` 进行变更

## 启动

### 训练

```bash
python main.py train
```

额外参数

-  `--resume` 继续上次的模型进行训练
-  `--no-gpu` 不使用 GPU 进行训练

### 测试

```bash
python main.py test
```

额外参数

-  `--no-gpu` 不使用 GPU 计算嵌入
-  `--use-cache` 使用之前已经计算过的嵌入缓存
