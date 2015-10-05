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
