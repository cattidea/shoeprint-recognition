# Usage

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

或者逐一进行安装，`tensorflow` 与 `tensorflow-gpu` 存一即可，版本号为 `1.13`

### 文件配置

可按照 `config.json` 对文件进行配置，默认配置如下

```
.
├── data
│   ├── cache
│   │   ├── data_loader
│   │   ├── data_set.h5
│   │   └── indices.json
│   ├── ckpt
│   ├── simple_graph
│   │   ├── C151224000005
│   │   ├── C151224000012
│   │   └── ...
│   ├── test
│   │   ├── C151224000012
│   │   └── ...
│   ├── train
│   │   ├── train_2
│   │   │   ├── C151224000012
│   │   │   └── ...
│   │   ├── train_剪切图
│   │   └── train_原图
│   ├── txt
│   │   └── 训练.txt
│   ├── result.txt
│   └── 索引.txt
├── docs
├── scripts
├── utils
├── config.json
└── main.py
```

## 训练

```bash
python main.py train
```

## 恢复模型继续训练

```bash
python main.py train --resume
```

## 测试

```bash
python main.py test
```
