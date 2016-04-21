0.4.2
---

* Renamed Python module so rather than `from pyscript.pyscript import ...` it is `from wekapyscript import ...`
* Renamed package to WekaPyScript, to avoid confusion
* Now works on latest build of WEKA

0.4.1
---

* Filter now has `-stdout` flag.
* `y` should now not exist in `args` if class attribute is not set (or if it is ignored).

0.4.0
---

* Filter system has changed - instead of process() returning an ARFF, it simply returns args.
* Add ignore class functionality (`ignore-class`).
* Fix bug in `instance_to_string` where it was assumed instances were not regression ones.

0.3.0
---

* Filters have now been implemented.
* Classifier and filter classes satisfy base unit tests.

0.2.1
---

* Can now choose to save the script in the model using the `-save` flag.

0.2.0
---

* Added Python 3 support.
* Added `uses` decorator to prevent non-essential arguments from being passed.
* Fixed nasty bug where imputation, binarisation, and standardisation would not actually
  be applied to test instances.
* GUI in WEKA now displays the exception as well.
* Fixed bug where single quotes in attribute values could mess up args creation.
* ArffToPickle now recognises class index option and arguments.
* Fix nasty bug where filters were not being saved and were made from scratch from test data.

0.1.1
---

* ArffToArgs gets temporary folder in a platform-independent way, instead of assuming /tmp/.
* Can now save args in ArffToPickle using `save`.

0.1.0
---

* Initial release.
