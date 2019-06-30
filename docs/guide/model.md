# Model

## 输入的处理

根据对样本图片大小的统计，得到平均高宽比 `2.79`，使用比较小的拟合比例 `11:4` 即`2.75`，故将图片 `resize` 到 `(132, 48)`

另外，参考大连海事大学的张弛学长的《基于卷积神经网络的鞋印图像分类算法研究》 $\href{#references}{^{[1]}}$ 文中提到的数据扩增方式对数据进行扩增，暂实现以下扩增方式

-  镜像扩增（水平镜像）
-  旋转扩增（顺时针 5、逆时针 5）
-  平移扩增（上下左右 30 像素）
-  椒盐噪声扩增（使用 0.01 密度的椒盐噪声）
-  随机遮挡扩增（40 个 30 \* 30 的块遮挡）
-  区域遮挡扩增（下半部一个水平遮挡条）

## 基本网络结构

基本网络结构参考 `vgg-16` $\href{#references}{^{[2]}}$ 的基本结构（卷积采用 `"same"` 只增加通道数，池化对图片大小进行缩减）

另外在两篇 Paper$\href{#references}{^{[1][3]}}$ 中都有提到 `maxout` $\href{#references}{^{[4]}}$ ，经测试，其效果确实优于 `ReLU` 激活函数，结合 `dropout` 会有着比较好的效果，故在全连接层采用 `maxout` 进行激活，全连接层参考 `vgg-16` 的卷积层对结点的缩减方式，结点的缩减仅在 `maxout` 处进行

> maxout 具体实现参考 $\href{#references}{^{[7]}}$ 与 `tf.contrib.layers.maxout` 源码

| layer   | size-in      | kernel                  | size-out       |
| ------- | ------------ | ----------------------- | -------------- |
| CONV1   | (132, 48, 1) | (3, 3, 8), s=1, "same"  | (132, 48, 8)   |
| POOL1   | (132, 48, 8) | (2, 2), s=2             | (66, 24, 8)    |
| CONV2   | (66, 24, 8)  | (3, 3, 16), s=1, "same" | (66, 24, 16)   |
| POOL2   | (66, 24, 16) | (2, 2), s=2             | (33, 12, 16)   |
| CONV3   | (33, 12, 16) | (3, 3, 32), s=1, "same" | (33, 12, 32)   |
| POOL3   | (33, 12, 32) | (3, 3), s=3             | (11, 4, 32)    |
| Flatten | (11, 4, 32)  |                         | 1408           |
| FC1     | 1408         | maxout, p=2             | 1024 $\to$ 512 |
| FC2     | 512          | maxout, p=2             | 512 $\to$ 256  |
| FC3     | 128          | maxout, p=2             | 256 $\to$ 128  |

另外，在最后对输出的 128 维向量进行 `l2` 归一化，能够使嵌入值满足同一分布

经测试，CONV 的 BN 层会严重影响计算速度，效果尚不佳，故本版本未采用，待有更充裕的时间与计算资源再做尝试

## loss 的计算

![triplet-loss](../Images/triplet-loss.jpg)

loss 采用 $Triplet Loss$ $\href{#references}{^{[5]}}$ ，也即 $L(A, P, N) = max(|| f(A) - f(P) ||^2 - || f(A) - f(N)||^2 + \alpha, 0)$ ，具体实现参考 $\href{#references}{^{[6]}}$ ，但是该方法对三元组的挑选并非在线评估选取，故三元组的在线选取参考了 $\href{#references}{^{[8]}}$

> 这里跳了很多坑，最终总算得到一个能用的模型，理论与实践的距离更近了一步

::: tip
关于图标

图标是在确定选题之前（因为四个选题都是图片识别，就想到了这样一个从众多图片中识别到某一个形象）就已经基本确定的，无意之中居然与 triplet-loss 中的图如此相似，也许这就是缘分吧~
:::

## 测试

根据网络计算的嵌入对原图以及待判定范围图片进行编码，根据原图与待判定范围嵌入之间的距离，可知距离原图最近的图片即目标图片，另外可对原图进行扩增进而进行综合比对，获得更高的准确率

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
