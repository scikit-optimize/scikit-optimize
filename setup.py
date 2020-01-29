try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import skopt

VERSION = skopt.__version__

setup(name='scikit-optimize',
      version=VERSION,
      description='Sequential model-based optimization toolbox.',
      long_description=open('README.rst').read(),
      url='https://scikit-optimize.github.io/',
      license='BSD',
      author='The scikit-optimize contributors',
      packages=['skopt', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process'],
      install_requires=['joblib', 'pyaml', 'numpy', 'scipy>=0.14.0',
                        'scikit-learn>=0.19.1'],
      extras_require={
        'plots':  ["matplotlib"]
        }
      )
