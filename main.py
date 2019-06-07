import matplotlib.pyplot as plt
import numpy as np
import random
import sys

from utils.data import data_import, test_data_import, get_determine_scope, get_shoeprint_arrays
from utils.train import train
from utils.test import test
from utils.imager import plot


# determine_scope = get_determine_scope(action_type="train")
# shoeprint_arrays = get_shoeprint_arrays(0, action_type="test")


# with open("data/索引2.txt", "w", encoding="utf8") as f:
#     for name, v in determine_scope.items():
#         if name in shoeprint_arrays:
#             f.write(name + "\t")
#             ls = list(v)
#             random.shuffle(ls)
#             f.write("\t".join(ls))
#             f.write("\n")

if sys.argv[1] == "train":
    train()
elif sys.argv[1] == "test":
    test()

#### test plot

# test_data_set_scope, test_data_set_origin, labels = test_data_import(debug=True)


# plt.imshow(np.reshape(test_data_set_origin[0], (78, 30)), cmap='gray')
# plt.show()
# plt.imshow(np.reshape(test_data_set_scope[0][labels[0][1]], (78, 30)), cmap='gray')
# plt.show()
# plt.imshow(np.reshape(test_data_set_scope[0][1], (78, 30)), cmap='gray')
# plt.show()
# plt.imshow(np.reshape(test_data_set_scope[0][2], (78, 30)), cmap='gray')
# plt.show()

#### train plot

# data_set = data_import()

# plot(data_set['X_dev_set'][0][0])
# plot(data_set['X_dev_set'][1][0])
# plot(data_set['X_dev_set'][2][0])

# plot(data_set['X_dev_set'][0][1])
# plot(data_set['X_dev_set'][1][1])
# plot(data_set['X_dev_set'][2][1])
