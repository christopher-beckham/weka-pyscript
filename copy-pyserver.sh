#!/bin/bash

MODE=$1

if [ $MODE == "new" ]; then
    echo "Copying modified py"
    cp scripts/pyServer.py $WEKA_HOME/packages/wekaPython/resources/py/pyServer.py
else
    echo "Copying default py"
    cp scripts/pyServerDefault.py $WEKA_HOME/packages/wekaPython/resources/py/pyServer.py
fi