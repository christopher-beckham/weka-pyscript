[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.33855.svg)](http://dx.doi.org/10.5281/zenodo.33855)

PyScriptClassifier
===

This package allows users to construct classifiers and filters with Python scripts for WEKA, given that the
script conforms to an expected structure. Get started with your first classifier [here](https://github.com/chrispy645/weka-pyscript/wiki/Getting-started)!

The tech report can be downloaded [here](http://www.cs.waikato.ac.nz/pubs/wp/2015/uow-cs-wp-2015-02.pdf).

Installation
---

This package requires the following:

* The latest and greatest version of WEKA. The nightly developer snapshot can be downloaded [here](http://www.cs.waikato.ac.nz/~ml/weka/snapshots/weka_snapshots.html).
* The [wekaPython](http://weka.sourceforge.net/packageMetaData/wekaPython/index.html) package written by Mark Hall. This package is actually a wrapper for Scikit-Learn, but it has code that makes it possible to interact with Python scripts.
* An installation of Python 2.7 with libraries installed such as Numpy and Pandas. The easiest (and safest) way to get these is to download the [Anaconda](http://continuum.io/downloads) distribution, since it comes with many essential packages preloaded.
* [ant](http://ant.apache.org/) to be able to build the package.
* Java 8, but 7 could probably work too.
* (Optional) [Theano](https://github.com/Theano/Theano) to be able to run the linear regression example.

When you have installed both of these, ensure that `wekaPython.jar` is in your `$CLASSPATH` variable. This .jar can be found in the `$WEKA_HOME/packages/wekaPython/` directory. I also assume that you have `weka.jar` in your classpath variable as well.

Now, download this Git repo, cd into the directory and run the following:

```
ant clean # if you have built the package previously
ant make_package -Dpackage=pyScript
cd dist
java weka.core.WekaPackageManager -install-package pyScript.zip
```

If the package installed successfully, you should now be able to run it from WEKA, either from the command-line or the GUI. A quick way to check if the classifier can be invoked is to simply run

```
java weka.Run .PyScriptClassifier
```

and see if WEKA recognises it. You should get an error like "Weka exception: No training file and no object input file given.".

Also make sure to install the `pyscript` Python module by running:

```
python setup.py install
```

Examples
---

#### Linear regression

Run a linear regressor on the diabetes dataset.

```
java weka.Run .PyScriptClassifier \
  -script scripts/linear-reg.py \
  -standardize \
  -t datasets/diabetes_numeric.arff -c last -no-cv
```

We can pass custom arguments in, and in this script two custom arguments can be specified to override the default values: `alpha` (the learning rate), and `epsilon` (early stopping criterion).

```
java weka.Run .PyScriptClassifier \
  -script scripts/linear-reg.py \
  -standardize \
  -args "alpha=0.001;epsilon=1e-6" \
  -t datasets/diabetes_numeric.arff -c last -no-cv
```

#### ZeroR

We can also run ZeroR on a nominal dataset such as Iris.

```
java weka.Run .PyScriptClassifier \
  -script scripts/zeror.py \
  -t datasets/iris.arff -c last -no-cv
```

#### Random forest

A Scikit-Learn random forest can be trained, passing in an argument `num_trees` which specifies how many trees should be used in the ensemble (this is a required argument and is not optional). To do a 10-fold cross-validation on `iris.arff` using 30 trees, we run:

```
java weka.Run .PyScriptClassifier \
  -script scripts/scikit-rf.py \
  -args "num_trees=30" \
  -t datasets/iris.arff
```

#### Standardise filter

We can also write Python scripts that act as filters. Here, we apply zero-mean unit-variance (ZMUV) standardisation to all numeric attributes in the data:

```
java weka.Run .PyScriptFilter \
  -script scripts/standardise.py \
  -i datasets/iris.arff \
  -c last
```
