import numpy as np
from sklearn.tree import DecisionTreeClassifier

do_use_weka = True
try:
	use_weka
except:
	do_use_weka = False

print "X_train", X_train.shape
print "y_train", y_train.shape

print "X_test", X_test.shape
print "y_test", y_test.shape

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

preds = clf.predict_proba(X_test).tolist()