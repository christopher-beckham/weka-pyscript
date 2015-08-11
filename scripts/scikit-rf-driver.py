import numpy as np
import gzip
import cPickle as pickle
import imp

f = gzip.open("../datasets/diabetes.pkl.gz.train", "rb")
args = pickle.load(f)
args["num_trees"] = 10
f.close()

cls = imp.load_source("cls", "scikit-rf.py")
model = cls.train(args)

args["X_test"] = args["X_train"]
preds = cls.test(args, model)

for pred in preds:
    print pred