import theano
from theano import tensor as T
import numpy as np
import gzip
import cPickle as pickle
from pyscript.pyscript import ArffToArgs, load_pkl

x = ArffToArgs()
x.set_input("../datasets/iris.arff")
x.set_class_index("last")
args = x.get_args()
x.close()

# ---

def list_to_indices(arr):
    dd = dict()
    for i in range(0, len(arr)):
        dd[ arr[i] ] = i
    return dd

def vector_to_string(vector, attributes, attr_values):
    vector = vector.tolist()
    string_vector = []
    for i in range(0, len(vector)):
        if attributes[i] in attr_values:
            string_vector.append( str(attr_values[ attributes[i] ][ int(vector[i]) ] ) )
        else:
            string_vector.append( str( vector[i] ) )
    return ",".join(string_vector)
    
def get_header(relation_name, attributes, attr_types, attr_values):
    header = []
    header.append("@relation %s" % relation_name)
    for attribute in attributes:
        if attribute in attr_values:
            header.append( "@attribute %s {%s}" % (attribute, ",".join(attr_values[attribute]) ) )
        else:
            header.append( "@attribute %s numeric" % attribute )
    header.append("@data")
    return "\n".join(header)

# ---

X_train = args["X_train"]
y_train = args["y_train"]
X_train = np.column_stack((X_train, y_train))

attr_values = args["attr_values"]
attr_types = args["attr_types"]
attributes = args["attributes"]
relation_name = args["relation_name"]

attr_to_idx = list_to_indices(attributes)
#print vector_to_string( X_train[0], attributes, attr_values )

print get_header(relation_name, attributes, attr_types, attr_values)
for inst in X_train:
    print vector_to_string(inst, attributes, attr_values)