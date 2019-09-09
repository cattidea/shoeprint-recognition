# 指南

## 要解决的问题

题目源于“恒悦杯”鞋印花纹图像类别判定挑战赛，现有的鞋底花纹图像检索算法返回的是排序后的类别，后续还需要人工从中找出与对应的花纹种属类别，人工判定就会消耗大量的人力和时间，所以研发鞋底花纹图像的自动类别判定算法意义重大

所以，给定一个数据集为鞋印样本数据集，其中包含若干（4537）种鞋印的样本数据，每种鞋一个鞋印图，该图相对完整、清晰

另有一个数据集为待判定的鞋印数据，该数据集内的鞋印图相对模糊且完整性没有保证，由一个文本文件记录每一个鞋印使用图片检索算法生成的相似度排序前 50 的类别，我们的任务是从这 50 个里面找出它真正的类别，该数据共有 500 类，每类 1-15 张图片（注意，测试集数据并不包含于这 500 类中）

## 问题分析

由于测试集与训练集（、开发集）的类别完全不同，那么简单的 Softmax 就是完全不可行的了，很容易想到使用相似度进行判定，最先想到的就是 FaceNet 中的 **triplet loss** 模型，该模型可以使 Anchor 与 Positive 之间的距离不断减小，同时使 Anchor 与 Negative 之间的距离不断增大，但是该模型有着收敛慢且容易过拟合的问题，但是在比较大的数据集下该模型还是比较适合的

既然测试集的类别与训练集（、开发集）不同，那么训练集与开发集的划分也应该按照类别进行划分，这样开发集的准确率才能大概表征测试集的准确率

> 在 triplet loss 的实现过程中踩了很多很多坑，起初使用离线生成的三元组进行训练，结果训练集很容易过拟合，在开发集上的表现就是什么都没有学到，准确率接近于瞎猜（1/50），后来采用了**在线生成**方才解决该问题，但是当前准确率依然不高（比较好的模型大概稳定在 45%）

## 开发环境

主体功能已在以下环境下测试~

1. 主 coding 环境

   -  CPU: Intel 4200U
   -  GPU: NVIDIA GTX950M
      -  CUDA: 9.0
   -  Memory: 8GB
   -  OS: Windows10
   -  Python: 3.6.7
      -  tensorflow-gpu 1.13.1
      -  (详见 `requirements.txt`)

2. 主 training 环境

   -  CPU: Intel 7500
   -  GPU: NVIDIA GTX1050
      -  CUDA: 10.0
   -  Memory: 8GB
   -  OS: Windows10
   -  Python: 3.6.7
      -  tensorflow-gpu 1.14.0

3. 简易 testing 环境（·视频速览·环境）

   -  硬件同环境 1
   -  OS: Deepin15.11
   -  Python: 3.5.3
      -  tensorflow 1.14.0