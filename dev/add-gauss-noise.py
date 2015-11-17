from pyscript.pyscript import ArffToArgs, get_header, vector_to_string
import numpy as np

def train(args):
    return ""

def filter(args, model):
    X = args["X"]
    y = args["y"]
    mean = args["mean"] if "mean" in args else 0
    sd = args["sd"] if "sd" in args else 1
    args["attributes"].remove(args["class"])
    args["attributes"].append( "attr_" + str(len(args["attributes"])-1) )
    args["attributes"].append(args["class"])
    buf = [get_header(args)]
    for i in range(0, X.shape[0]):
        vector = X[i].tolist()
        vector.append( np.random.normal(mean,sd) )
        vector.append( y[i].tolist()[0] )
        buf.append( vector_to_string(vector, args) )
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
