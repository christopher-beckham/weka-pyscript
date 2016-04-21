from __future__ import print_function
from wekapyscript import ArffToArgs, \
    get_header, instance_to_string
import numpy as np

def train(args):
    X_train = args["X_train"]
    means = []
    sds = []
    attr_types = args["attr_types"]
    attributes = args["attributes"]
    for i in range(0, X_train.shape[1]):
        if attr_types[ attributes[i] ] == "numeric":
            means.append( np.nanmean(X_train[:,i]) )
            sds.append(
                np.nanstd(X_train[:,i], ddof=1) )
        else:
            means.append(None)
            sds.append(None)
    return (means, sds)

def process(args, model):
    X = args["X"]
    attr_types = args["attr_types"]
    attributes = args["attributes"]
    means, sds = model
    for i in range(0, X.shape[1]):
        if attr_types[ attributes[i] ] == "numeric":
            X[:,i] = (X[:,i] - means[i]) / sds[i]
    return args
        
if __name__ == '__main__':
    x = ArffToArgs()
    x.set_input("../datasets/iris.arff")
    x.set_class_index("last")
    args = x.get_args()
    print (args.keys())
    x.close()
    model = train(args)
    args["X"] = args["X_train"]
    args["y"] = args["y_train"]
    process(args, model)
