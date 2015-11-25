from __future__ import print_function
from subprocess import call
import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import sys
import time
import tempfile
import shutil
import numpy as np

def load_pkl(filename):
    f = gzip.open(filename)
    args = pickle.load(f)
    f.close()
    return args

class uses(object):
    def __init__(self, used_args):
        self.used_args = used_args
        self.defaults = set([
            "X_train", "y_train", "X_test", "class_type", "relation_name",
            "attributes", "attr_values", "class", "num_classes", "attr_types"
        ])
    def __call__(self, f):
        def wrapped_f(*args):
            args_variable = args[0]
            for var in args_variable:
                if var not in self.used_args and var not in self.defaults:
                    raise ValueError("This classifier does not use the non-default variable: '%s'" % var)
            return f(*args)
        return wrapped_f

class ArffToArgs(object):
    def __init__(self):
        self.input = ""
        self.output = ""
        self.class_index = "last"
        self.standardize = ""
        self.binarize = ""
        self.impute = ""
        self.debug = ""
        self.arguments = ""

    def set_standardize(self, b):
        assert isinstance(b, bool)
        self.standardize = "-standardize" if b else ""
    def set_binarize(self, b):
        assert isinstance(b, bool)
        self.binarize = "-binarize" if b else ""
    def set_impute(self, b):
        assert isinstance(b, bool)
        self.impute = "-impute" if b else ""
    def set_input(self, filename):
        self.input = filename
    def set_debug(self, b):
        assert isinstance(b, bool)
        self.debug = "-debug" if b else ""
    def set_class_index(self, class_index):
        self.class_index = class_index
    def set_arguments(self, arguments):
        self.arguments = arguments

    def get_args(self):
        if self.input == "" or self.class_index == "":
            raise ValueError("Make sure you have used set_input, and set_class_index at least")
        self.output = tempfile.gettempdir() + os.path.sep + "%s_%f.pkl.gz" % ( os.path.basename(self.input), time.time() )
        self.output = self.output.replace("\\", "\\\\") # for windows
        driver = ["java", "weka.Run", "weka.pyscript.ArffToPickle",
            "-i", self.input, "-o", self.output ]
        if self.class_index != None:
            driver.append("-c")
            driver.append(self.class_index)
        driver.append("-args")
        driver.append(self.arguments)
        driver.append(self.standardize)
        driver.append(self.binarize)
        driver.append(self.impute)
        driver.append(self.debug)
        sys.stderr.write("%s\n" % " ".join(driver))
        result = call(driver)
        if result != 0:
            raise Exception("Error - Java call returned a non-zero value")
        else:
            return load_pkl(self.output)

    def save(self, filename):
        shutil.move(self.output, filename)

    def close(self):
        try:
            os.remove(self.output)
        except OSError:
            pass

def get_header(args):

    relation_name = args["relation_name"]
    attributes = args["attributes"]
    attr_types = args["attr_types"]
    attr_values = args["attr_values"]
    
    header = []
    header.append("@relation %s" % relation_name)
    for attribute in attributes:
        if attribute in attr_values:
            header.append( "@attribute %s {%s}" % (attribute, ",".join(attr_values[attribute]) ) )
        else:
            header.append( "@attribute %s numeric" % attribute )
    header.append("@data")
    return "\n".join(header)



def instance_to_string(x, y, args):
    attributes = args["attributes"]
    attr_values = args["attr_values"]
    string_vector = []
    for i in range(0, len(x)):
        if np.isnan(x[i]):
            string_vector.append("?")
        else:
            if attributes[i] in attr_values:
                string_vector.append( str(attr_values[ attributes[i] ][ int(x[i]) ] ) )
            else:
                string_vector.append( str( x[i] ) )
    if y != None:
        if np.isnan(y[0]):
            string_vector.append("?")
        else:
            string_vector.append( attr_values["class"][int(y[0])] )
    return ",".join(string_vector)

if __name__ == '__main__':

    x = ArffToArgs()
    x.set_input("../datasets/iris.arff")
    #x.set_output("/tmp/iris.pkl.gz")
    x.set_standardize(True)
    x.set_binarize(True)
    x.set_impute(True)
    x.set_class_index("last")
    x.set_arguments("a='\\'foo\\'';b='bar';c=0.001")
    print(x.get_args().keys())
    x.close()
