===========
Development
===========

The library is still experimental and under heavy development. Checkout
the `next
milestone <https://github.com/scikit-optimize/scikit-optimize/milestone/7>`__
for the plans for the next release or look at some `easy
issues <https://github.com/scikit-optimize/scikit-optimize/issues?q=is%3Aissue+is%3Aopen+label%3AEasy>`__
to get started contributing.

The development version can be installed through:

.. code-block:: bash

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -e.

Run all tests by executing ``pytest`` in the top level directory.

To only run the subset of tests with short run time, you can use ``pytest -m 'fast_test'`` (``pytest -m 'slow_test'`` is also possible). To exclude all slow running tests try ``pytest -m 'not slow_test'``.

This is implemented using pytest `attributes <https://docs.pytest.org/en/latest/mark.html>`__. If a tests runs longer than 1 second, it is marked as slow, else as fast.

All contributors are welcome!


Making a Release
~~~~~~~~~~~~~~~~

The release procedure is almost completely automated. By tagging a new release
travis will build all required packages and push them to PyPI. To make a release
create a new issue and work through the following checklist:

* update the version tag in ``__init__.py``
* update the version tag mentioned in the README
* check if the dependencies in ``setup.py`` are valid or need unpinning
* check that the ``CHANGELOG.md`` is up to date
* did the last build of master succeed?
* create a `new release <https://github.com/scikit-optimize/scikit-optimize/releases>`__
* ping `conda-forge <https://github.com/conda-forge/scikit-optimize-feedstock>`__

Before making a release we usually create a release candidate. If the next
release is v0.X then the release candidate should be tagged v0.Xrc1 in
``__init__.py``. Mark a release candidate as a "pre-release"
on GitHub when you tag it.
