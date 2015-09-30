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

    def get_args(self):
        if self.input == "" or self.class_index == "":
            raise ValueError("Make sure you have used set_input, and set_class_index at least")
        self.output = tempfile.gettempdir() + os.path.sep + "%s_%f.pkl.gz" % ( os.path.basename(self.input), time.time() )
        self.output = self.output.replace("\\", "\\\\") # for windows
        driver = ["java", "weka.Run", "weka.pyscript.ArffToPickle",
            "-i", self.input, "-o", self.output, "-c", self.class_index, self.standardize, self.binarize,
            self.impute, self.debug
        ]
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

if __name__ == '__main__':

    x = ArffToArgs()
    x.set_input("../datasets/iris.arff")
    #x.set_output("/tmp/iris.pkl.gz")
    x.set_standardize(True)
    x.set_binarize(True)
    x.set_impute(True)
    x.set_class_index("last")
    x.get_args()
    x.close()