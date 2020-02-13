try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.5 is needed.
    import __builtin__ as builtins

# This is a bit (!) hackish: we are setting a global variable so that the
# main skopt __init__ can detect if it is being loaded by the setup
# routine
builtins.__SKOPT_SETUP__ = True

import skopt

VERSION = skopt.__version__

setup(name='scikit-optimize',
      version=VERSION,
      description='Sequential model-based optimization toolbox.',
      long_description=open('README.rst').read(),
      url='https://scikit-optimize.github.io/',
      license='BSD 3-clause "New" or "Revised License"',
      author='The scikit-optimize contributors',
      packages=['skopt', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process'],
      install_requires=['joblib', 'pyaml', 'numpy', 'scipy>=0.14.0',
                        'scikit-learn>=0.19.1'],
      extras_require={
        'plots':  ["matplotlib"]
        }
      )
