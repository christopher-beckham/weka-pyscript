For this package to work, the `wekaPython` package must be installed. The `pyServer.py` file that comes
with that package must be replaced by the one specified in this repo (i.e. run `copy-pyserver.sh`) because
of issues related to running `exec` in Python. For more information on that particular issue:

http://stackoverflow.com/questions/24712192/why-are-module-level-variables-in-a-python-exec-inaccessible

tl;dr - don't run `exec (script, globals, locals)` - just do `exec in globals`.