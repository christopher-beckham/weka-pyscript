import theano
from theano import tensor as T
import numpy as np

def train(args):
    w = theano.shared( np.ones( (args["num_attributes"]-1, 1) ), name='w', borrow=True)
    b = theano.shared( 1.0, name='b', borrow=True)

    x = T.dmatrix('x')
    y = T.dmatrix('y')
    out = T.dot(x, w) + b

    #X_train = np.asarray(args["X_train"], dtype="float32")
    #y_train = np.asarray(args["y_train"], dtype="float32")
    X_train = args["X_train"]
    y_train = args["y_train"]

    cost = (T.sum((out - y)**2)) / args["num_instances"]

    g_w = T.grad(cost=cost, wrt=w)
    g_b = T.grad(cost=cost, wrt=b)

    alpha=0.001
    updates = [ (w, w - alpha*g_w), (b, b - alpha*g_b) ]

    train = theano.function([x, y], outputs=cost, updates=updates)

    prev_loss = train(X_train, y_train)
    for epoch in range(0, 10000):
        this_loss = train(X_train, y_train)
        if abs(this_loss - prev_loss) < args["epsilon"]:
            break
        prev_loss = this_loss

    return [ w.get_value(), b.get_value() ]

def describe(args, weights):
    return "linear model"

def test(args, weights):
    
    w = theano.shared( np.ones( (args["num_attributes"], 1) ), name='w', borrow=True)
    b = theano.shared( 1.0, name='b', borrow=True)
    w.set_value( weights[0] )
    b.set_value( weights[1] )

    x = T.dmatrix('x')
    out = T.dot(x, w) + b
    pred = theano.function([x], out)

    #X_test = np.asarray(args["X_test"], dtype="float32")
    X_test = args["X_test"]

    return pred(X_test).tolist()


if __name__ == '__main__':
    print "Supported args:"
    print "alpha (learning rate)"
    print "  e.g. alpha=0.01"
    print "epsilon (stopping criterion)"
    print "  e.g. epsilon=0.00001"