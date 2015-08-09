import numpy as np
import gzip
import cPickle as pickle
import imp

f = gzip.open("../datasets/diabetes.pkl.gz.train", "rb")
args = pickle.load(f)
args["alpha"] = 0.1
f.close()

cls = imp.load_source("cls", "logistic-reg.py")
cls.train(args)