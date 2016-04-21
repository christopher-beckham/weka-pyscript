from wekapyscript import ArffToArgs, get_header, instance_to_string
import numpy as np

def train(args):
    return ""

def process(args, model):
    X = args["X"]
    mean = args["mean"] if "mean" in args else 0
    sd = args["sd"] if "sd" in args else 1
    if "class" in args:
        args["attributes"].remove( args["class"] )
    args["attributes"].append( "attr_" + str(len(args["attributes"])-1) )
    if "class" in args:
        args["attributes"].append( args["class"] )
    X_new = []
    for i in range(0, X.shape[0]):
        vector = X[i].tolist()
        vector.append( np.random.normal(mean,sd) )
        X_new.append(vector)
    X_new = np.asarray(X_new, dtype="float32")
    args["X"] = X_new
    return args
        
if __name__ == '__main__':
    x = ArffToArgs()
    x.set_input("../datasets/iris.arff")
    x.set_class_index("last")
    args = x.get_args()
    x.close()
    model = train(args)
    args["X"] = args["X_train"]
    print process(args, model)
