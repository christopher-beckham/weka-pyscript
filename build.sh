#!/bin/bash

ant make_package -Dpackage=pyScript
cd dist
java weka.core.WekaPackageManager -install-package pyScript.zip
cd ../datasets
java weka.gui.explorer.Explorer diabetes_numeric.arff