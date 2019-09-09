# FAQ

::: danger Q1:

为什么运行之后效果非常差，根本达不到文档里所述的效果？

:::

::: tip A1:

效果差很可能是缓存不对应的问题造成的，当原数据变更后，如果不及时清除缓存的话，程序会优先加载缓存数据，这样自然会引发错误，需要删除的缓存主要有

-  图片数据的各种缓存，在 `cache_loader` 中
-  图片数据的二级缓存，有一个 `h5` 文件和 `json` 文件
-  样本库的嵌入缓存，该缓存在样本图库或者模型变更的情况下都会造成数据变更

:::

---

::: danger Q2:

为什么程序运行后处理图片时会报错？

:::

::: tip A2:

首先请确保路径配置正确，另外，请尽量不要在包含图片的路径中出现中文，否则 cv2 可能无法读取图片

:::

---

::: danger Q3:

为什么会出现 OOM？应该如何避免？

:::

::: tip A3:

首先看**测试**的情况，测试时仅仅会使用训练好的网络计算嵌入，而计算嵌入使用的是分批计算，每次计算， `emb_step` 张图片的嵌入值，如果发生 OOM ，很可能是你的==显存/内存==不足以放下相应大小的图片，可以通过减小 `emb_step` 以避免该问题，相反地，如果你的配置允许，可以通过提高 `emb_step` 以提高计算速度

> `emb_step` 可在 `config.train.emb_step` 处调整

如果是**训练**的话，如果发生了 OOM 可能是==内存==无法容纳足够的数据，如果真的发生该问题，请减少部分扩增方式

另外，训练过程如果有过多数据同时通过神经网络的话，也会引起==内存/显存== OOM，但是该问题的引发部位可能有两个，一个是计算嵌入以获得三元组的时候，该问题解决方案参考上面测试情况下的解决方案，另一种就是三元组通过神经网络的时候，该问题可通过降低 `class_per_batch` 、 `show_per_class` 、 `img_per_shoe` 解决，但是都不要低于 3 ，否则三元组生成器可能无法正常运行（过低的数值也会使得训练效果也会大打折扣）

> 上面的三个参数也都可以在 `config.train` 中进行修改

:::