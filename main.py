import matplotlib.pyplot as plt
import numpy as np
import random
import sys

from utils.data import data_import, test_data_import, get_determine_scope, get_shoeprint_arrays, get_simple_arrays
from utils.train import train
from utils.test import test
from utils.imager import plot

# get_simple_arrays(0)
# get_shoeprint_arrays(0, "train")
# data_import()

# determine_scope = get_determine_scope(action_type="train")
# shoeprint_arrays, shoeprint_map = get_shoeprint_arrays(0, action_type="test")


# with open("data/索引2.txt", "w", encoding="utf8") as f:
#     for name, v in determine_scope.items():
#         if name in shoeprint_map:
#             f.write(name + "\t")
#             ls = list(v)
#             random.shuffle(ls)
#             f.write("\t".join(ls))
#             f.write("\n")

#### test plot

# img_arrays, test_data_map = test_data_import()

#for name in test_data_map:
#    if input(">"):
#        break
#    plot(img_arrays[test_data_map[name]["index"]])
 #   plot(img_arrays[test_data_map[name]["scope_indices"][test_data_map[name]["label"]]])
  #  plot(img_arrays[test_data_map[name]["scope_indices"][0]])
#    plot(img_arrays[test_data_map[name]["scope_indices"][1]])

#### train plot

# data_set = data_import()
# X_imgs = data_set["X_imgs"]
# X_indices_train_set = data_set["X_indices_train_set"]

# print(X_indices_train_set.shape)

# for i in range(100):
#     if input(">"):
#         break
#     plot(X_imgs[X_indices_train_set[0][i]])
#     plot(X_imgs[X_indices_train_set[1][i]])
#     plot(X_imgs[X_indices_train_set[2][i]])

if sys.argv[1] == "train":
    train()
elif sys.argv[1] == "test":
    test()
