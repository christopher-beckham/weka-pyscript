from __future__ import print_function

import theano
from theano import tensor as T
import numpy as np
import gzip

try:
    import cPickle as pickle
except ImportError:
    import pickle

from wekapyscript import ArffToArgs, uses

@uses(["alpha", "epsilon"])
def train(args):

    X_train = args["X_train"]
    y_train = args["y_train"]

    # let w be a p*1 vector, and b be the intercept
    num_attributes = X_train.shape[1]
    w = theano.shared( np.zeros( (num_attributes, 1) ), name='w')
    b = theano.shared( 1.0, name='b')

    # let x be a n*p matrix, and y be a n*1 matrix
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    # prediction is simply xw + b
    out = T.dot(x, w) + b

    # cost function is mean squared error
    num_instances = X_train.shape[0]
    cost = (T.sum((out - y)**2)) / num_instances
    # compute gradient of cost w.r.t. w and b
    g_w = T.grad(cost=cost, wrt=w)
    g_b = T.grad(cost=cost, wrt=b)

    alpha = 0.01 if "alpha" not in args else args["alpha"]
    epsilon = 1e-6 if "epsilon" not in args else args["epsilon"]

    updates = [ (w, w - alpha*g_w), (b, b - alpha*g_b) ]

    train = theano.function([x, y], outputs=cost, updates=updates)

    prev_loss = train(X_train, y_train)
    for epoch in range(0, 100000):
        this_loss = train(X_train, y_train)
        print(this_loss)
        if abs(this_loss - prev_loss) < epsilon:
            break
        if np.isnan(this_loss):
            raise Exception("Loss is NaN! Have you made sure to " +
                            "standardise the attributes before-hand?\n" +
                            "Also ensure only numeric attributes are present.")
        prev_loss = this_loss

    return [ w.get_value(), b.get_value() ]

def describe(args, weights):
    coefs = weights[0].flatten()
    intercept = weights[1]
    st = "f(x) = \n"
    for i in range(0, len(coefs)):
        st += "  " + args["attributes"][i] + "*" + str(coefs[i]) + " +\n"
    st += "  " + str(intercept)
    return st

def test(args, weights):
    X_test = args["X_test"]
    num_attributes = X_test.shape[1]

    w = theano.shared( np.zeros( (num_attributes, 1) ), name='w')
    b = theano.shared( 1.0, name='b' )
    w.set_value( weights[0] )
    b.set_value( weights[1] )

    x = T.dmatrix('x')
    out = T.dot(x, w) + b
    pred = theano.function([x], out)

    X_test = args["X_test"]
    return pred(X_test).tolist()


if __name__ == '__main__':

    x = ArffToArgs()
    x.set_input("../datasets/diabetes_numeric.arff")
    x.set_class_index("last")
    x.set_standardize(True)
    args = x.get_args()
    print(args)
    args["alpha"] = 0.01
    args["epsilon"] = 1e-6
    model = train(args)
    print(describe(args, model))
    x.close()
