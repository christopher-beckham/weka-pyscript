import theano
from theano import tensor as T
import numpy as np
import gzip
import cPickle as pickle

def train(args):

    # let w be a p*1 vector, and b be the intercept
    w = theano.shared( np.zeros( (args["num_attributes"], 1) ), name='w')
    b = theano.shared( 1.0, name='b')

    # let x be a n*p matrix, and y be a n*1 matrix
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    # prediction is simply xw + b
    out = T.dot(x, w) + b

    X_train = args["X_train"]
    y_train = args["y_train"]

    # cost function is mean squared error
    cost = (T.sum((out - y)**2)) / args["num_instances"]
    # compute gradient of cost w.r.t. w and b
    g_w = T.grad(cost=cost, wrt=w)
    g_b = T.grad(cost=cost, wrt=b)

    alpha = args["alpha"]
    updates = [ (w, w - alpha*g_w), (b, b - alpha*g_b) ]

    train = theano.function([x, y], outputs=cost, updates=updates)

    prev_loss = train(X_train, y_train)
    for epoch in range(0, 100000):
        this_loss = train(X_train, y_train)
        print this_loss
        if abs(this_loss - prev_loss) < args["epsilon"]:
            break
        prev_loss = this_loss

    return [ w.get_value(), b.get_value() ]

def describe(args, weights):
    coefs = weights[0].flatten()
    intercept = weights[1]
    st = "f(x) = \n"
    for i in range(0, len(args["attributes"])):
        st += "  " + args["attributes"][i] + "*" + str(coefs[i]) + " +\n"
    st += "  " + str(intercept)
    return st


def test(args, weights):    
    w = theano.shared( np.zeros( (args["num_attributes"], 1) ), name='w')
    b = theano.shared( 1.0, name='b' )
    w.set_value( weights[0] )
    b.set_value( weights[1] )

    x = T.dmatrix('x')
    out = T.dot(x, w) + b
    pred = theano.function([x], out)

    X_test = args["X_test"]
    return pred(X_test).tolist()


if __name__ == '__main__':
    print "Supported args:"
    print "alpha (learning rate)"
    print "  e.g. alpha=0.01"
    print "epsilon (stopping criterion)"
    print "  e.g. epsilon=0.00001"