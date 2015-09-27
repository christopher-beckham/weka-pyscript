from pyscript.pyscript import ArffToArgs

def get_args(filename, sd, bn, im):
    x = ArffToArgs()
    x.set_input(filename)
    x.set_class_index("last")

    x.set_impute(im)
    x.set_binarize(bn)
    x.set_standardize(sd)

    args = x.get_args()
    x.close()
    return args

"""
print "normal..."
print get_args("../datasets/various.arff", False, False, False)["X_train"]

print "standardisation..."
print get_args("../datasets/various.arff", True, False, False)["X_train"]

print "binarize..."
print get_args("../datasets/various.arff", False, True, False)["X_train"]

print "impute..."
print get_args("../datasets/various.arff", False, False, True)["X_train"]
"""

print "all..."
args = get_args("../datasets/various.arff", True, True, True)
print args["X_train"]
print args["attr_types"]
