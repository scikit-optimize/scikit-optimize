try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import versioneer


setup(name='ProcessOptimizer',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Sequential model-based optimization toolbox (forked from scikit-optimize)',
      long_description=open('README.rst').read(),
      url='https://github.com/bytesandbrains/ProcessOptimizer',
      license='BSD',
      author='Bytes and Brains',
      packages=['ProcessOptimizer', 'ProcessOptimizer.learning', 'ProcessOptimizer.optimizer', 'ProcessOptimizer.space',
                'ProcessOptimizer.learning.gaussian_process'],
      install_requires=['pyaml', 'numpy', 'scipy>=0.14.0',
                        'scikit-learn>=0.19.1'],
      extras_require={
        'plots':  ["matplotlib"]
        }
      )
