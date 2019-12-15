# 一些实验性的想法

有些是在实现过程中发现并不是十分有效所以弃用的方案，还有一些是比赛之后学习了新的一些 ML 以及 DL 的知识进而产生的一些新想法

## Mask 机制

也即求出图片的残缺模板 [$^{[1]}$](#references)，在网络的接近输出层进行过滤，将残缺部位的特征过滤掉，以提高残缺图像的识别率，但是由于效果并不好，弃用

## 更深的网络深度

虽然网络深度的加深往往有着更好的效果，但是更深的网络使得每个 batch 会更小，模型更难收敛，经 ModelV3、ModelV5 的测试，取消该方案

## 输出前的可分离卷积使用多个卷积核

也即原来每个通道卷积为一个标量，现在卷积为一个向量，之后再进行后续处理，ModelV3 保留了一个简单实现，但由于效果不好，后续没时间在其他模型测试

## 优化 Batch Normalization <font color=#000080><sup>TODO</sup></font>

### 使用 GN 替代 BN

根据 GN 的效果描述，它可以用于解决 BN 在少量 `mini-batch size` 下效果差的问题，很符合现状

### 使用 SELU 替代 ReLU

利用 SELU 的特性实现 BN 的效果

## 在线扩增数据 <font color=#000080><sup>TODO</sup></font>

使用在线扩增方式对数据进行扩增，但这必然会增加数据处理的时间，需要运算足够快的 CPU 或者显存足够大的 GPU

## 更改 Loss 计算方式 <font color=#000080><sup>TODO</sup></font>

使用 Auto Encoder 或者 Siamese Network 以降低网络的复杂度，如果在资源有限的情况下应该有着更好的效果，但是在资源足够的情况下并不是最佳选择

# References {docsify-ignore}

1. 卜凡杰. [鞋印图像特征提取及检索方法研究](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201901&filename=1018717266.nh)
