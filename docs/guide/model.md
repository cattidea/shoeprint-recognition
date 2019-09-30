# 模型

## 基本网络结构

本程序在神经网络的结构与深度上进行了若干次尝试，从基本的 `vgg-16` 开始，进而使用 `inception_v2`$\href{#references}{^{[10]}}$ 模块，之后通过研读 CVPR 论文获悉到前几年刚刚提出的新模型，比如 `Xception`$\href{#references}{^{[12]}}$ 、 `ResNeXt`$\href{#references}{^{[13]}}$ 、`MobileNet`$\href{#references}{^{[14]}}$ 等等，经过测试，最终保留了 `Inception_v2` 、 `Xception` 与本人自行尝试设计的一个网络（暂时命名为 `Nnet_v1`），其中以 `Nnet_v1` 效果最佳，但由于经验不足与测试比较少，并不代表该网络真的超越了其他网络（在想 Peach）

其中 `Xception` 与 `Nnet_v1` 均保留了基本版本与深度版本，但是由于两个深度版本（V3 和 V5）表现并不是很好，所以后续并不会提及这两个深度版本

除了网络结构，在激活函数的选择上也做了一定的取舍，考虑到 `ReLU` 是一个性能与效果都非常好的激活函数，一般网络的激活均使用了 `ReLU` ，此外，在两篇 Paper$\href{#references}{^{[1][3]}}$ 中都有提到 `maxout` $\href{#references}{^{[4]}}$ ，经测试，其效果确实优于 `ReLU` 激活函数，但由于其应用不广，仅在部分网络输出之前以及 `Nnet_v1` 使用

> maxout 具体实现参考 $\href{#references}{^{[7]}}$ 与 `tf.contrib.layers.maxout` 源码

## 网络流程图

::: tip

默认使用 ModelV4 （Nnet_v1），修改训练模型需要修改 `model.Model` 的父类，修改测试模型需要修改 `model_dir` （默认 `data/ckpt/`）相关配置

:::

### Inception

也即 ModelV1

| layer        | size-in       | kernel                   | size-out      |
| ------------ | ------------- | ------------------------ | ------------- |
| CONV1        | (129, 49, 1)  | (3, 3, 32), s=1, "same"  | (129, 49, 32) |
| POOL1        | (129, 49, 32) | (3, 3), s=2              | (65, 25, 32)  |
| INCEPTION_2a | (65, 25, 32)  | (3, 3, 64), s=1, "same"  | (65, 25, 64)  |
| INCEPTION_2b | (65, 25, 64)  | (3, 3, 64), s=1, "same"  | (65, 25, 64)  |
| POOL2        | (65, 25, 64)  | (3, 3), s=2              | (33, 13, 64)  |
| INCEPTION_3a | (33, 13, 64)  | (3, 3, 128), s=1, "same" | (33, 13, 128) |
| INCEPTION_3b | (33, 13, 128) | (3, 3, 128), s=1, "same" | (33, 13, 128) |
| POOL3        | (33, 13, 128) | (3, 3), s=2              | (17, 7, 128)  |
| INCEPTION_4a | (17, 7, 128)  | (3, 3, 256), s=1, "same" | (17, 7, 256)  |
| INCEPTION_4b | (17, 7, 256)  | (3, 3, 256), s=1, "same" | (17, 7, 256)  |
| INCEPTION_4c | (17, 7, 256)  | (3, 3, 256), s=1, "same" | (17, 7, 256)  |
| POOL4        | (17, 7, 256)  | (3, 3), s=2              | (9, 4, 256)   |
| MAXOUT       | (9, 4, 256)   | k=2                      | (9, 4, 128)   |
| DW_CONV      | (9, 4, 128)   | (9, 4), s=1, "valid"     | (1, 1, 128)   |
| Flatten      | (1, 1, 128)   |                          | (128, )       |
| l2_norm      | (128, )       |                          | (128, )       |

::: tip

-  所有激活函数均选择 ReLU
-  使用可分离卷积代替全局池化（参考 MobileNet）
-  两个模块之间有残差连接，如果两个模块 shape 不一样，则使用 $1 \times 1$ 卷积进行调节（参考 Inception_v4 $\href{#references}{^{[11]}}$）
-  所有池化均使用 $3 \times 3$ 卷积核并且 $stride = 2$、$padding="same"$ （参考 Xception 论文）
-  所有卷积层均使用 Batch_Normalization（参考 Inception_v2）

:::

### Xception

也即 ModelV2

| layer            | size-in       | kernel                   | size-out      |
| ---------------- | ------------- | ------------------------ | ------------- |
| CONV1            | (129, 49, 1)  | (3, 3, 32), s=1, "same"  | (129, 49, 32) |
| POOL1            | (129, 49, 32) | (3, 3), s=2              | (65, 25, 32)  |
| Separable_CONV2a | (65, 25, 32)  | (3, 3, 64), s=1, "same"  | (65, 25, 64)  |
| Separable_CONV2b | (65, 25, 64)  | (3, 3, 128), s=1, "same" | (65, 25, 128) |
| POOL2            | (65, 25, 64)  | (3, 3), s=2              | (33, 13, 64)  |
| Separable_CONV3a | (33, 13, 128) | (3, 3, 128), s=1, "same" | (33, 13, 128) |
| Separable_CONV3b | (33, 13, 128) | (3, 3, 128), s=1, "same" | (33, 13, 128) |
| Separable_CONV3c | (33, 13, 128) | (3, 3, 128), s=1, "same" | (33, 13, 128) |
| Separable_CONV4a | (33, 13, 128) | (3, 3, 128), s=1, "same" | (33, 13, 128) |
| Separable_CONV4b | (33, 13, 128) | (3, 3, 128), s=1, "same" | (33, 13, 128) |
| POOL3            | (33, 13, 128) | (3, 3), s=2              | (17, 7, 128)  |
| Separable_CONV5a | (17, 7, 128)  | (3, 3, 256), s=1, "same" | (17, 7, 256)  |
| Separable_CONV5a | (17, 7, 256)  | (3, 3, 256), s=1, "same" | (17, 7, 256)  |
| POOL4            | (17, 7, 256)  | (3, 3), s=2              | (9, 4, 256)   |
| MAXOUT           | (9, 4, 256)   | k=2                      | (9, 4, 128)   |
| DW_CONV          | (9, 4, 128)   | (9, 4), s=1, "valid"     | (1, 1, 128)   |
| Flatten          | (1, 1, 128)   |                          | (128, )       |
| l2_norm          | (128, )       |                          | (128, )       |

::: tip

-  所有激活函数均选择 ReLU
-  使用可分离卷积代替全局池化（参考 MobileNet）
-  两个模块之间有残差连接，如果两个模块 shape 不一样，则使用 $1 \times 1$ 卷积进行调节（参考 Xception 论文）
-  所有池化均使用 $3 \times 3$ 卷积核并且 $stride = 2$、$padding="same"$ （参考 Xception 论文）
-  所有卷积层均使用 Batch_Normalization（参考 Inception_v2）

:::

### Nnet_v1

也即 ModelV4

| layer   | size-in      | params                   | size-out     |
| ------- | ------------ | ------------------------ | ------------ |
| Nnet_1  | (129, 49, 1) | (3, 3, 16), s=2, "same"  | (65, 25, 16) |
| Nnet_2a | (65, 25, 16) | (3, 3, 16), s=1, "same"  | (65, 25, 16) |
| Nnet_2b | (65, 25, 16) | (3, 3, 32), s=2, "same"  | (33, 13, 32) |
| Nnet_3a | (33, 13, 32) | (3, 3, 32), s=1, "same"  | (33, 13, 32) |
| Nnet_3b | (33, 13, 64) | (3, 3, 64), s=1, "same"  | (33, 13, 64) |
| Nnet_4a | (33, 13, 32) | (3, 3, 64), s=1, "same"  | (33, 13, 64) |
| Nnet_4b | (33, 13, 64) | (3, 3, 128), s=2, "same" | (17, 7, 128) |
| Nnet_5a | (17, 7, 128) | (3, 3, 128), s=1, "same" | (17, 7, 128) |
| Nnet_5b | (17, 7, 128) | (3, 3, 128), s=2, "same" | (9, 4, 128)  |
| DW_CONV | (9, 4, 128)  | (9, 4), s=1, "valid"     | (1, 1, 128)  |
| Flatten | (1, 1, 128)  |                          | (128, )      |
| l2_norm | (128, )      |                          | (128, )      |

::: warning 模块设计的一些细节

虽然网络设计简单了，事实上 Nnet_1 模块内部做了不少事情，首先陈述以下事实：

-  Average_pooling 等价于卷积核全为 1 的可分离卷积，所以可分离卷积会比 Average_pooling 做的更好（参考 MobileNet）
-  $f \times f$ 的普通卷积等价于 $f \times f$ 的可分离卷积 + $1 \times 1$ 的普通卷积，且效果更好（参考 Xception）
-  $3 \times 3$卷积等价于最外两圈全是零的 $5 \times 5$ 卷积，也等价于两次 $3 \times 3$ 卷积且第二个卷积核中间为 $1$ 外圈为 $0$ ，所以，大的感受野总是优于小的感受野，同时大的感受野是可以被多层小的感受野所替换的（参考 Inception_v2）
-  BN 使得层与层之间解耦
-  Max_pooling 与 Maxout 都有着比较好的效果
-  ResNets 能够保留前层特征用于后层的组合，以学习到更丰富的特征
-  Inception Xception 以及 ResNeXt 都对通道进行了一定的分离，取得了可观的效果

基于以上事实，每个 Nnet_v1 模块对输入的 Feature Map 做以下变换

1. 首先进行 $1 \times 1$ 卷积降低通道数，形成瓶颈层以降低运算量，之后对瓶颈层进行 $3 \times 3$ 可分离卷积，同时从输入计算残差，将残差与可分离卷积结果在通道轴上拼接

2. 如果该层需要对 size 进行缩减的话，对刚刚拼接的结果进行两次 $1 \times 1$ 卷积，两个结果分别进行最大池化与可分离卷积，然后在通道上进行拼接

3. 如果该层需要非线性激活的话，对刚刚拼接的结果进行两次 $1 \times 1$ 卷积，对其中一个结果进行 maxout 激活（k = 4），然后与另一个结果在通道轴上进行拼接

4. 最后对结果进行 Batch_Normalization

:::

::: tip 为什么这样会有效？我猜想有下原因

-  更多的超参数是由网络自行学习的，比如是不是需要残差、是选用最大池化还是可分离卷积、需不需要激活、怎么激活
-  即使该模块进行了激活，也有可能有一部分未激活的特征流入下一层，两层卷积可以提高感受野
-  可能有一部分残差通过通道流入了下一层，低层次特征与高层次特征相互组合以组成更丰富的特征

虽然想了不少，但是很多都是自己的一些猜想，暂时无从也无时间进行更加专业的考证，有机会会继续优化并使用 ImageNet 测试实际效果

:::

## 一些实验性的想法

### Mask 机制

也即求出图片的残缺模板 $\href{#references}{^{[9]}}$，在网络的接近输出层进行过滤，将残缺部位的特征过滤掉，以提高残缺图像的识别率，但是由于效果并不好，弃用

### 更深的网络深度

虽然网络深度的加深往往有着更好的效果，但是更深的网络使得每个 batch 会更小，模型更难收敛，经 ModelV3、ModelV5 的测试，取消该方案

### 输出前的可分离卷积使用多个卷积核

也即原来每个通道卷积为一个标量，现在卷积为一个向量，之后再进行后续处理，ModelV3 保留了一个简单实现，但由于效果不好，后续没时间在其他模型测试

### 使用 GN 替代 BN

> 刚刚了解，尚未尝试，但是根据描述可能会有很好的优化效果

GN 可以用于解决 BN 在少量 mini-batch size 下效果差的问题

## loss 的计算

![triplet-loss](../Images/triplet-loss.jpg)

loss 采用 $Triplet Loss$ $\href{#references}{^{[5]}}$ ，也即 $L(A, P, N) = max(|| f(A) - f(P) ||^2 - || f(A) - f(N)||^2 + \alpha, 0)$ ，具体实现参考 $\href{#references}{^{[6]}}$ ，但是该方法对三元组的挑选并非在线评估选取，故三元组的在线选取参考了 $\href{#references}{^{[8]}}$

# References

1. 张弛. 基于卷积神经网络的鞋印图像分类算法研究
2. Simonyan, Zisserman 2015. Very deep convolutional networks for large-scale image recognition
3. Florian Schroff, Dmitry Kalenichenko, James Philbin. Schroff FaceNet A Unified 2015 CVPR paper
4. Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, Yoshua Bengio 2013. Maxout Networks
5. Taigman et. al., 2014. DeepFace closing the gap to human level performance
6. [triplet-loss-mnist](https://github.com/SpikeKing/triplet-loss-mnist)
7. [How to use maxout activation function in tensorflow?](https://stackoverflow.com/questions/39954793/how-to-use-maxout-activation-function-in-tensorflow)
8. [FaceNet（一）---Triplet Loss](https://blog.csdn.net/baidu_27643275/article/details/79222206)
9. 卜凡杰. 鞋印图像特征提取及检索方法研究
10.   [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)
11.   [Inception-v4, Inception-ResNet](https://arxiv.org/pdf/1602.07261.pdf)
12.   [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
13.   [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
14.   [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
