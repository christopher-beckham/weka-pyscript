from subprocess import call
import gzip
import cPickle as pickle
import os
import sys
import time

def load_pkl(filename):
    f = gzip.open(filename)
    args = pickle.load(f)
    f.close()
    return args

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
    def set_output(self, filename):
        self.output = filename
    def set_debug(self, b):
        assert isinstance(b, bool)
        self.debug = "-debug" if b else ""
    def set_class_index(self, class_index):
        self.class_index = class_index

    def get_args(self):
        if self.input == "" or self.class_index == "":
            raise ValueError("Make sure you have used set_input, and set_class_index at least")
        if self.output == "":
            self.output = "/tmp/%s_%f.pkl.gz" % ( os.path.basename(self.input), time.time() )
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
        else:
            return load_pkl(self.output)

    def close(self):
        call(["rm", self.output])

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