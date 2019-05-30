import matplotlib.pyplot as plt
import numpy as np

from utils.data import data_import, test_data_import
from utils.train import train
from utils.test import test

train()
# test()

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

# plt.imshow(np.reshape(data_set['X_dev_set'][0][0], (78, 30)), cmap='gray')
# plt.show()
# plt.imshow(np.reshape(data_set['X_dev_set'][0][1], (78, 30)), cmap='gray')
# plt.show()
# plt.imshow(np.reshape(data_set['X_dev_set'][0][2], (78, 30)), cmap='gray')
# plt.show()
# plt.imshow(np.reshape(data_set['X_dev_set'][0][3], (78, 30)), cmap='gray')
# plt.show()
