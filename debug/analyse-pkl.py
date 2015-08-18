import numpy as np
import gzip
import cPickle as pickle
import imp
import sys

f = gzip.open(sys.argv[1], "rb")
args = pickle.load(f)
f.close()

for key in args:
    if key in ["X_train", "y_train", "X_test", "y_test"]:
        print key, "\t\t\t", args[key].shape
    else:
        print key, "\t\t\t", args[key]