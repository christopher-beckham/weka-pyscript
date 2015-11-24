from pyscript.pyscript import ArffToArgs, get_header, instance_to_string
import numpy as np

def train(args):
    X_train = args["X_train"]
    means = []
    sds = []
    for i in range(0, X_train.shape[1]):
        means.append( np.nanmean(X_train[:,i]) )
        sds.append( np.nanstd(X_train[:,i], ddof=1) )
    return (means, sds)

def filter(args, model):
    X = args["X"]
    y = args["y"]
    means = model[0]
    sds = model[1]
    for i in range(0, X.shape[1]):
        X[:,i] = (X[:,i] - means[i]) / sds[i]
    header = get_header(args)
    buf = [header]
    for i in range(0, X.shape[0]):
        buf.append( instance_to_string(X[i], y[i], args) )
    return "\n".join(buf)
        
if __name__ == '__main__':
    x = ArffToArgs()
    x.set_input("../datasets/iris.arff")
    x.set_class_index("last")
    args = x.get_args()
    x.close()
    model = train(args)
    args["X"] = args["X_train"]
    args["y"] = args["y_train"]
    print filter(args, model)
