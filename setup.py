try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='ProcessOptimizer',
      version='0.2.2',
      description='Sequential model-based optimization toolbox (forked from scikit-optimize)',
      url='https://github.com/bytesandbrains/ProcessOptimizer',
      license='BSD',
      author='Bytes and Brains',
      packages=['ProcessOptimizer', 'ProcessOptimizer.learning', 'ProcessOptimizer.optimizer', 'ProcessOptimizer.space',
                'ProcessOptimizer.learning.gaussian_process'],
      install_requires=['pyaml', 'numpy', 'matplotlib', 'scipy>=0.14.0',
                        'scikit-learn>=0.19.1', 'bokeh']
      )
