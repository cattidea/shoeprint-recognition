import numpy as np

from config_parser.config import MARGIN


class BatchAll():
    """ Batch all 策略 在线生成 Triplet """
    def __init__(self, model, indices, class_per_batch, shoe_per_class, img_per_shoe,
                    img_arrays, sess):
        self.model = model
        self.indices = indices
        self.class_per_batch = class_per_batch
        self.shoe_per_class = shoe_per_class
        self.img_per_shoe = img_per_shoe
        self.img_arrays = img_arrays
        self.sess = sess
        self.alpha = MARGIN
        self.start_index = 0
        self.shadow_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_index >= self.shadow_index:
            self.shadow_index = self.start_index
            shoeprints, nrof_shoes_per_class, self.start_index = self.sample_shoeprint(self.indices, self.start_index, self.class_per_batch, self.shoe_per_class, self.img_per_shoe)
            embeddings = self.model.compute_embeddings(self.img_arrays[shoeprints], self.sess)
            triplets = self.select_triplets(embeddings, shoeprints, nrof_shoes_per_class, self.class_per_batch, self.img_per_shoe, self.alpha)
            return self.shadow_index, triplets
        else:
            raise StopIteration


    @staticmethod
    def sample_shoeprint(data_set, start_index, class_per_batch, shoe_per_class, img_per_shoe):
        """ 抽取一个 batch 所需的鞋印
        ``` python
        [
            <idx01>, <idx02>, ...
        ]
        ```
        """
        nrof_shoes = class_per_batch * shoe_per_class
        nrof_classes = len(data_set)
        img_per_shoe_origin = len(data_set[0][0])

        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)

        shoeprints = []
        nrof_shoes_per_class = []

        while len(shoeprints) < nrof_shoes:
            # print("sample_shoeprint {}/{} ".format(len(shoeprints), nrof_shoes), end='\r')
            class_index = class_indices[start_index]
            # 某一类中鞋印的总数量
            nrof_shoes_in_class = len(data_set[class_index])
            if nrof_shoes_in_class > 1:
            # if True:
                shoe_indices = np.arange(nrof_shoes_in_class)
                np.random.shuffle(shoe_indices)
                # 该类中需要抽取鞋印的数量
                nrof_shoes_from_class = min(nrof_shoes_in_class, shoe_per_class, nrof_shoes-len(shoeprints))
                idx = shoe_indices[: nrof_shoes_from_class]
                # 随机选取一定量的扩增图
                img_indices = np.random.choice(img_per_shoe_origin, img_per_shoe, replace=False)
                shoeprints += [np.array(data_set[class_index][i])[img_indices] for i in idx]
                nrof_shoes_per_class.append(nrof_shoes_from_class)

            start_index += 1
            start_index %= nrof_classes

        assert len(shoeprints) == nrof_shoes
        return np.reshape(shoeprints, (nrof_shoes * img_per_shoe, )), nrof_shoes_per_class, start_index

    @staticmethod
    def select_triplets(embeddings, shoeprints, nrof_shoes_per_class, class_per_batch, img_per_shoe, alpha):
        """ 选择三元组 """
        emb_start_idx = 0
        triplets = []

        for i in range(len(nrof_shoes_per_class)):
            # print("select_triplets {}/{} ".format(i, class_per_batch), end='\r')
            nrof_shoes = int(nrof_shoes_per_class[i])
            if nrof_shoes <= 1:
                continue

            # 某个鞋
            for j in range(0, nrof_shoes*img_per_shoe, img_per_shoe):
                a_offset = np.random.randint(img_per_shoe) # 同图偏移
                a_idx = emb_start_idx + j + a_offset
                neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), axis=-1)
                # 将本类鞋距离设为无穷，不作 negative
                neg_dists_sqr[emb_start_idx: emb_start_idx+nrof_shoes*img_per_shoe] = np.inf

                for k in range(j+img_per_shoe, nrof_shoes*img_per_shoe, img_per_shoe):
                    p_offset = np.random.randint(img_per_shoe)
                    p_idx = emb_start_idx + k + p_offset
                    pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                    # 由于 neg_dist 中有 NaN ，故会有 RuntimeWarning
                    all_neg = np.where(neg_dists_sqr-pos_dist_sqr < alpha)[0]
                    nrof_random_negs = all_neg.shape[0]

                    if nrof_random_negs > 0:
                        # 如果存在满足条件的 neg ，则随机挑选一个
                        rnd_idx = np.random.randint(nrof_random_negs)
                        n_idx = all_neg[rnd_idx]
                        triplets.append((shoeprints[a_idx], shoeprints[p_idx], shoeprints[n_idx]))

                    # neg_loss = neg_dists_sqr - pos_dist_sqr - alpha
                    # n_idx = np.argmin(neg_loss)
                    # if neg_loss[n_idx] < 0:
                    #     triplets.append((shoeprints[a_idx], shoeprints[p_idx], shoeprints[n_idx]))

            emb_start_idx += nrof_shoes * img_per_shoe

        np.random.shuffle(triplets)
        return triplets
