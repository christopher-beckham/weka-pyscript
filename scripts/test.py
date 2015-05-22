import numpy as np
from sklearn.tree import DecisionTreeClassifier

print "X_train", X_train.shape
print "y_train", y_train.shape

print "X_test", X_test.shape
print "y_test", y_test.shape

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

preds = clf.predict_proba(X_test).tolist()