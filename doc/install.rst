.. _installation-instructions:

============
Installation
============

scikit-optimize supports Python 3.5 or newer.

The newest release can be installed via pip:

.. code-block:: bash

    $ pip install scikit-optimize

or via conda:

.. code-block:: bash

    $ conda install -c conda-forge scikit-optimize

The newest development version of scikit-optimize can be installed by:

.. code-block:: bash

    $ pip install git+https://github.com/scikit-optimize/scikit-optimize.git

Development version
~~~~~~~~~~~~~~~~~~~

The library is still experimental and under heavy development.
The development version can be installed through:

.. code-block:: bash

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -r requirements.txt
    python setup.py develop

Run the tests by executing `pytest` in the top level directory.