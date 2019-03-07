# ProcessOptimizer

ProcessOptimizer is a fork of scikit-optimize, that focuses on optimizing real world processes, like chemistry or baking.
For examples on use checkout out https://scikit-optimize.github.io/.

## Installation
ProcessOptimizer can be installed using "pip install ProcessOptimizer"
The repository and examples can be found at https://github.com/bytesandbrains/ProcessOptimizer/
ProcessOptimizer can also be installed by running pip install -e. in top directory of the downloaded repository.

## PyPi
If you have not packaged before check out https://packaging.python.org/tutorials/packaging-projects/
To upload a new version to PyPi do the following:
* Change version number in setup.py
* run `python3 setup.py sdist bdist_wheel`
* run `python3 -m twine upload dist/*`