from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from wekapyscript import ArffToArgs, uses

@uses(["num_trees"])
def train(args):
    X_train = args["X_train"]
    y_train = args["y_train"].flatten()
    num_trees = args["num_trees"] if "num_trees" in args else 10
    rf = RandomForestClassifier(n_estimators=num_trees, random_state=0)
    rf = rf.fit(X_train, y_train)
    return rf

def describe(args, model):
    return "RandomForestClassifier with %i trees" % model.n_estimators

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