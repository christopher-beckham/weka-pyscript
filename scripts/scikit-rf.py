from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from pyscript.pyscript import ArffToArgs

def train(args):
    X_train = args["X_train"]
    y_train = args["y_train"].flatten()
    rf = RandomForestClassifier(n_estimators=args["num_trees"])
    rf = rf.fit(X_train, y_train)
    return rf

def describe(args, model):
    return "RandomForestClassifier with %i trees" % args["num_trees"]

def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()

if __name__ == '__main__':
    x = ArffToArgs()
    x.set_input("../datasets/iris.arff")
    x.set_class_index("last")
    args = x.get_args()
    args["num_trees"] = 10
    rf = train(args)
    print( describe(args, rf) )
    x.close()