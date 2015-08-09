import numpy as np
from numpy.random import random
import sys

EPS = 1e-15

def init_weights(p):
    return np.asarray([random() for x in range(0, p)], dtype="float32")

def hypothesis(x, theta):
    W = theta[0]
    b = theta[1]
    return 1.0 / (1.0 + np.exp( -1 * (np.dot(x, W) + b)) )

def cost(X, y, weights):
    hypotheses = hypothesis(X, weights)
    s = 0
    assert len(hypotheses) == len(y)
    for i in range(0, len(hypotheses)):
        s += ( y[i]*np.log(hypotheses[i]+EPS) + (1-y[i])*np.log(1-hypotheses[i]+EPS) )
    return s / len(y)

def train(args):
    X_train = args["X_train"]
    y_train = args["y_train"].flatten()
    p = X_train.shape[1]
    m = X_train.shape[0]

    W = init_weights(p)
    b = 0

    # gradient descent
    alpha = args["alpha"]

    for epoch in range(0, 1000):

        new_W = init_weights(p)
        new_b = 0

        for j in range(0, len(W)):
            ss = 0
            for i in range(0, m):
                ss += ( hypothesis(X_train[i], [W,b]) - y_train[i] ) * X_train[i][j]
            new_W[j] = W[j] - (alpha/m)*ss

        ss = 0
        for i in range(0, m):
            ss += ( hypothesis(X_train[i], [W,b]) - y_train[i] )
        new_b = b - (alpha/m)*ss

        W = new_W
        b = new_b

        preds = np.asarray( hypothesis(X_train, [W,b]) >= 0.5, dtype="float32" )
        print float(np.sum(np.equal(preds, y_train))) / len(y_train) * 100



def test(args, model):
    pass

def describe(args, model):
    pass