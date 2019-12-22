# 指南

## 要解决的问题

> 题目源于“恒悦杯”鞋印花纹图像类别判定挑战赛特别赛道

由于现有的鞋底花纹图像检索算法返回的是排序后的类别，后续还需要人工从中找出与对应的花纹种属类别，人工判定就会消耗大量的人力和时间，所以研发鞋底花纹图像的自动类别判定算法意义重大

现给定一个数据集为鞋印样本数据集，其中包含若干（4537）种鞋印的样本数据，每种鞋一个鞋印图，该图相对完整、清晰

另有一个数据集为待判定的鞋印数据，该数据集内的鞋印图相对模糊且完整性没有保证，由一个文本文件记录每一个鞋印使用图片检索算法生成的相似度排序前 50 的类别，我们的任务是从这 50 个里面找出它真正的类别，该数据共有 500 类，每类 1-15 张图片（注意，测试集数据并不包含于这 500 类中）

## 问题分析

### 主要流程

由于测试集与训练集（、开发集）的类别完全不同，那么简单的 Softmax 就是完全不可行的了，很容易想到使用相似度进行判定，那么如何计算相似度呢？

“一切皆可 Embedding”，很容易就想到可以想到对图片计算一个 Embedding，根据 Embedding 在欧式空间的距离来评估两张图片的相似度，那么问题又来了，如何获得一个好的计算 Embedding 的模型？

我最先想到的就是 FaceNet 中的 **triplet loss** 模型，该模型可以使 Anchor 与 Positive 之间的距离不断减小，同时使 Anchor 与 Negative 之间的距离不断增大，但是该模型有着收敛慢且容易过拟合的问题，但是在比较大的数据集下该模型还是比较适合的

当然，在此问题上还有些其他模型也是合适的，比如 Siamese Network 或者 Auto-Encoder 都是可以尝试的选项，但我觉得在数据足够且资源富裕的条件下，Triplet loss 仍然是更好的选择

### 其他问题

既然测试集的类别与训练集（、开发集）不同，那么训练集与开发集的划分也应该按照类别进行划分，这样开发集的准确率才能大概表征测试集的准确率

另外，为了提高模型的泛化能力，需要对图片进行一定的扩增处理

## 开发环境

?> 环境很一般，我相信如果有更好的环境，很轻松就可以取得更好的效果

-  CPU: Intel 7500
-  Memory: 8GB
-  GPU: NVIDIA GTX1050
   -  Mermory: 2GB
   -  CUDA: 10.0
-  OS: Windows10
-  Python: 3.6.7
   -  tensorflow-gpu 1.14.0