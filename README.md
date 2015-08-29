[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.29050.svg)](http://dx.doi.org/10.5281/zenodo.29050)

PyScriptClassifier
===

This package allows users to construct classifiers with Python scripts for WEKA, given that the
script conforms to an expected structure.

Installation
---

This package requires the following:

* An installation of Python 2.7 with libraries installed such as Numpy and Pandas. The easiest (and safest) way to get these is to download the [Anaconda](http://continuum.io/downloads) distribution, as this is what I have used when I have developed this package.
* The [wekaPython](http://weka.sourceforge.net/packageMetaData/wekaPython/index.html) package written by Mark Hall. This package is actually a wrapper for Scikit-Learn, but it has code that makes it possible to interact with Python scripts.
* [ant](http://ant.apache.org/) to be able to build the package.
* Java 8, but 7 could probably work too.

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
java weka.Run weka.classifiers.pyscript.PyScriptClassifier
```

and see if WEKA recognises it. You should get an error like "Weka exception: No training file and no object input file given.".
