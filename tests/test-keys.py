from __future__ import print_function
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


args = get_args("../datasets/various.arff", True, True, True)
print (args.keys())
