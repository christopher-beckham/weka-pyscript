from collections import Counter
import numpy as np

def train(args):
    return Counter(args["y_train"].flatten(). \
        tolist()). \
        most_common()[0][0]

def describe(args, model):
    return "Majority class: %i" % model

def test(args, model):
    m = int(model)
    return [ np.eye( args["num_classes"] )[m]. \
        tolist() \
        for x in range(0, args["X_test"].shape[0]) ]
