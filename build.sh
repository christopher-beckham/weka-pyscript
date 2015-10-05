#!/bin/bash

ant clean
ant make_package -Dpackage=PyScriptClassifier
cd dist
java weka.core.WekaPackageManager -install-package PyScriptClassifier.zip
#cd ../datasets
#java weka.gui.explorer.Explorer diabetes_numeric.arff
