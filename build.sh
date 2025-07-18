#!/bin/bash

if [ -z $WEKA_HOME ]; then
    echo "WEKA_HOME is not set! Exiting"
    exit 1
fi

WEKA_PYTHON_JAR=$WEKA_HOME/packages/wekaPython/wekaPython.jar
if [ ! -f $WEKA_PYTHON_JAR ]; then
    echo "Cannot find ${WEKA_PYTHON_JAR}! Do you have the wekaPython package installed?"
    exit 1
fi

# WEKA must be in the classpath already
CLASSPATH=$CLASSPATH:$WEKA_PYTHON_JAR

ant clean
ant make_package -Dpackage=WekaPyScript
cd dist
java weka.core.WekaPackageManager -offline -install-package WekaPyScript.zip
