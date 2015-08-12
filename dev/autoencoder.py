import theano
from theano import tensor as T
import lasagne
import gzip
import cPickle as pickle
import numpy as np
import sys

f = gzip.open("../data/mnist10k.pkl.gz", "rb")
args = pickle.load(f)
f.close()

l_in = lasagne.layers.InputLayer( (None, args["num_attributes"]) )
l_noise = lasagne.layers.GaussianNoiseLayer(l_in)
l_hidden = lasagne.layers.DenseLayer(
    l_noise, num_units=50, nonlinearity=lasagne.nonlinearities.sigmoid
)
l_out = lasagne.layers.DenseLayer(l_hidden, num_units=args["num_attributes"])

# print network
layers = lasagne.layers.get_all_layers(l_out)
for i in range(0, len(layers)):
    print layers[i], lasagne.layers.get_output_shape(layers[i])

# ----

X_train = np.asarray( args["X_train"], dtype="float32" )

# constrain pixels to be between 0 and 1
for i in range(0, X_train.shape[0]):
   X_train[i] /= 255

input_var = T.fmatrix('x')

prediction = lasagne.layers.get_output(l_out, input_var)
hidden = lasagne.layers.get_output(l_hidden, input_var)
loss = lasagne.objectives.squared_error(prediction, input_var).mean()

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.rmsprop(
    loss, params, learning_rate=0.01
)

train_fn = theano.function([input_var], loss, updates=updates)
hidden_fn = theano.function([input_var], hidden)
predict_fn = theano.function([input_var], prediction)

batch_size = 128
for epoch in range(0, 10):
    print "Epoch: %i" % epoch
    avg_loss = []
    b = 0
    while True:
        X_subset = X_train[b*batch_size : (b+1)*batch_size]
        if X_subset.shape[0] == 0:
            break
        else:
            avg_loss.append( train_fn(X_subset) )
        b += 1
    print "  loss =", np.mean(avg_loss)

for vector in hidden_fn(X_train):
    print vector.tolist()