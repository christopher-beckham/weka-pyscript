from sklearn.ensemble import RandomForestClassifier

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