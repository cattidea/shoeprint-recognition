import sys

from utils.train import train
from utils.test import test

if sys.argv[1] == "train":
    train()
elif sys.argv[1] == "test":
    test()
