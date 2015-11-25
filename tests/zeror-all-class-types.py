from collections import Counter
import numpy as np

def train(args):
    if args["attr_types"][ args["class"] ] != "numeric":
        return Counter(args["y_train"].flatten(). \
                tolist()). \
                most_common()[0][0]
    else:
        return 0

def describe(args, model):
    return "Majority class: %i" % model

def test(args, model):
    if args["attr_types"][ args["class"] ] != "numeric":
        return [ np.eye( args["num_classes"] )[model]. \
                 tolist() \
                 for x in range(0, args["X_test"].shape[0]) ]
    else:
        return [ [0.0] for x in range(0, args["X_test"].shape[0]) ]
