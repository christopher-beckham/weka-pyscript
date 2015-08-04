import lasagne
import theano
from theano import tensor as T
from lasagne.objectives import categorical_crossentropy as x_ent
from lasagne.regularization import l2
from lasagne.regularization import regularize_network_params as reg
import sys
import numpy as np
import imp
import time
import os

import gzip
import cPickle as pickle

class Container(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

symbols = None
args = None
SEED = 0

def lenet():
    # https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    l_in = lasagne.layers.InputLayer( shape=(None, 1, 28, 28) )
    l_conv1 = lasagne.layers.Conv2DLayer(l_in, filter_size=(5,5), num_filters=20)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))
    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, filter_size=(5,5), num_filters=50)
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2))
    l_hidden = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform()
    )
    return l_out

def lenet_skinny():
    # https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    l_in = lasagne.layers.InputLayer( shape=(None, 1, 28, 28) )
    l_conv1 = lasagne.layers.Conv2DLayer(l_in, filter_size=(5,5), num_filters=20/2)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))
    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, filter_size=(5,5), num_filters=50/2)
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2))
    l_hidden = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=500/2,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform()
    )
    return l_out


def prepare():

    X = T.tensor4('X')
    y = T.ivector('y')

    output_layer = lenet_skinny()

    all_params = lasagne.layers.get_all_params(output_layer)

    loss_fn = x_ent

    objective = lasagne.objectives.Objective(output_layer, loss_function=loss_fn)
    loss = objective.get_loss(X, target=y) + args["lambda"]*reg(output_layer, l2)

    label_vector = output_layer.get_output(X)
    pred = T.argmax( label_vector, axis=1 )
    accuracy = T.mean( T.eq(pred,y) )

    return Container(
        { "X": X, "y": y, "output_layer": output_layer, "all_params": all_params,
        "objective": objective, "loss": loss, "label_vector": label_vector,
        "pred": pred, "accuracy": accuracy
        }
    )

def train(arg):
    global args, symbols, best_weights, SEED
    args = arg

    if "save" in args:
        f = gzip.open(args["save"], "wb")
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    X_train = args["X_train"]
    y_train = args["y_train"]

    X_train = np.asarray(X_train, dtype="float32")
    y_train = np.asarray(y_train.flatten(), dtype="int32")
    X_train = X_train.reshape( (X_train.shape[0], 1, 28, 28) )

    symbols = prepare()

    alpha = args["alpha"]
    momentum = 0.9

    if alpha != -1:
        if "rmsprop" not in args:
            updates = lasagne.updates.momentum(symbols.loss, symbols.all_params, alpha, momentum)
        else:
            updates = lasagne.updates.rmsprop(symbols.loss, symbols.all_params, alpha)
    else:
        updates = lasagne.updates.adagrad(symbols.loss, symbols.all_params, 1.0)

    iter_train = theano.function(
        [symbols.X, symbols.y],
        [symbols.label_vector, symbols.pred, symbols.loss, symbols.accuracy],
        updates=updates
    )

    sys.stderr.write(str(X_train.shape)+"\n")
    sys.stderr.write(str(y_train.shape)+"\n")

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    best_valid_accuracy = -1
    for e in range(0, args["epochs"]):

        np.random.seed(SEED)
        np.random.shuffle(X_train)
        np.random.seed(SEED)
        np.random.shuffle(y_train)

        SEED += 1
        np.random.seed(SEED)

        sys.stderr.write("Epoch #%i:\n" % e)
        batch_train_losses = []
        batch_train_accuracies = []
        batch_train_alt_losses = []
        for b in range(0, X_train.shape[0]):
            if b*bs >= X_train.shape[0]:
                break
            #sys.stderr.write("  Batch #%i (%i-%i)\n" % ((b+1), (b*bs), ((b+1)*bs) ))
            v, _, loss, acc = iter_train( X_train[b*bs : (b+1)*bs], y_train[b*bs : (b+1)*bs] )
            batch_train_losses.append(loss)
            batch_train_accuracies.append(acc)

        sys.stderr.write( "  train_loss, train_accuracy = %f, %f\n" % \
            (np.mean(batch_train_losses), np.mean(batch_train_accuracies)) )

    best_weights = lasagne.layers.get_all_param_values(symbols.output_layer)

    return best_weights

def describe(arg, weights):
    return "blank"

def test(arg, weights):

    args = arg

    symbols = prepare()
    lasagne.layers.set_all_param_values(symbols.output_layer, weights)
    iter_test = theano.function(
        [symbols.X],
        symbols.label_vector
    )

    args["X_test"] = np.asarray(args["X_test"], dtype="float32")

    if "batch_size" in args:
        bs = args["batch_size"]
    else:
        bs = 128

    X_test = args["X_test"]
    preds = iter_test(X_test).tolist()

    return preds