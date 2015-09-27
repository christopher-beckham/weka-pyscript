#!/bin/bash

# create a zeror model from iris.arff and save the model to /tmp/zeror.model
java weka.Run .PyScriptClassifier -script ../scripts/zeror.py -t ../datasets/iris.arff -no-cv -d /tmp/zeror.model

# use the model to predict on the same dataset (iris.arff)
java weka.Run .PyScriptClassifier -l /tmp/zeror.model -T ../datasets/iris.arff

# make a copy of the script somewhere else
cp ../scripts/zeror.py /tmp/zeror.py

# make a new model that references /tmp/zeror.py instead of scripts/zeror.py
java weka.Run .ChangeScriptPath -i /tmp/zeror.model -o /tmp/zeror_2.model -script "/tmp/zeror.py"

# use new model to predict iris.arff
java weka.Run .PyScriptClassifier -l /tmp/zeror_2.model -T ../datasets/iris.arff
