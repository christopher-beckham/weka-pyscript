from subprocess import call
import gzip
import cPickle as pickle
import os
import sys
import time

class ArffToPickle(object):
    def __init__(self):
        self.input = ""
        self.output = ""
        self.class_index = "last"
        self.standardize = ""
        self.binarize = ""
        self.impute = ""
        self.debug = ""
    def set_standardize(self, b):
        self.standardize = "-standardize" if b else ""
    def set_binarize(self, b):
        self.binarize = "-binarize" if b else ""
    def set_impute(self, b):
        self.impute = "-impute" if b else ""
    def set_input(self, filename):
        self.input = filename
    def set_output(self, filename):
        self.output = filename
    def set_debug(self, b):
        self.debug = "-debug" if b else ""
    def set_class_index(self, class_index):
        self.class_index = class_index

    def get_pkl(self):
        if self.output == "":
            output = "/tmp/%s_%f.pkl.gz" % ( os.path.basename(self.input), time.time() )
        else:
            output = self.output
        if self.input == "" or self.class_index == "":
            raise ValueError("Make sure you have used set_input, and set_class_index at least")
        driver = ["java", "weka.Run", "weka.pyscript.ArffToPickle",
            "-i", self.input, "-o", output, "-c", self.class_index, self.standardize, self.binarize, self.impute, self.debug
        ]
        #print " ".join(driver)
        result = call(driver)
        if result != 0:
            raise Exception("blahhh")
        else:
            f = gzip.open(output)
            args = pickle.load(f)
            f.close()
            call(["rm", output])
            return args


if __name__ == '__main__':

    x = ArffToPickle()
    x.set_input("../datasets/iris.arff")
    #x.set_output("/tmp/iris.pkl.gz")
    #x.set_standardize(True)
    #x.set_binarize(True)
    #x.set_impute(True)
    x.set_class_index("last")
    print x.get_pkl()