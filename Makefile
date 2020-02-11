# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= pytest
CTAGS ?= ctags

# skip doctests on 32bit python
BITS := $(shell python -c 'import struct; print(8 * struct.calcsize("P"))')

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist
	# TODO: Remove in when all modules are removed.
	$(PYTHON) sklearn/_build_utils/deprecated_modules.py

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(PYTEST) --showlocals -v skopt --durations=20
test-sphinxext:
	$(PYTEST) --showlocals -v doc/sphinxext/
test-doc:
ifeq ($(BITS),64)
	$(PYTEST) $(shell find doc -name '*.rst' | sort)
endif
test-code-parallel: in
	$(PYTEST) -n auto --showlocals -v skopt --durations=20

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) skopt --showlocals -v --cov=sklearn --cov-report=html:coverage
test-coverage-parallel:
	rm -rf coverage .coverage .coverage.*
	$(PYTEST) skopt -n auto --showlocals -v --cov=sklearn --cov-report=html:coverage

test: test-code test-sphinxext test-doc

trailing-spaces:
	find skopt -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R skopt

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 sklearn | grep -v __init__ | grep -v external
	pylint -E -i y skopt/ -d E1103,E0611,E1101

flake8-diff:
	./build_tools/circle/linting.sh
