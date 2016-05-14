from wekapyscript import ArffToArgs, get_header, instance_to_string
import numpy as np

def train(args):
    return ""

def process(args, model):
    X = args["X"]
    if "class" in args:
        args["attributes"].remove( args["class"] )
    new_attribute_idx = len(args["attributes"])-1
    args["attributes"].append("nominal_attr_%i" % new_attribute_idx)
    if "class" in args:
        args["attributes"].append( args["class"] )
    X_new = []
    for i in range(0, X.shape[0]):
        vector = X[i].tolist()
        rand = np.random.randint(0,3)
        if rand==0:
            vector.append(0)
        elif rand==1:
            vector.append(1)
        else:
            vector.append(2)
        X_new.append(vector)
    X_new = np.asarray(X_new, dtype="float32")

    # we need to say this new attribute is nominal
    args["attr_types"]["nominal_attr_%i" % new_attribute_idx] = "nominal"
    # what values does this attribute take?
    args["attr_values"]["nominal_attr_%i" % new_attribute_idx] = ["a", "b", "c"]

    args["X"] = X_new
    return args

#'attributes': ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class'], 'attr_values': {'class': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']}, 'class': 'class', 'num_classes': 3, 'attr_types': {'sepallength': 'numeric', 'petalwidth': 'numeric', 'sepalwidth': 'numeric', 'class': 'nominal', 'petallength': 'numeric'}}
        
if __name__ == '__main__':
    x = ArffToArgs()
    x.set_input("../datasets/iris.arff")
    x.set_class_index("last")
    args = x.get_args()
    x.close()
    model = train(args)
    args["X"] = args["X_train"]
    print process(args, model)