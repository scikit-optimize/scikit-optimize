try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='scikit-optimize',
      version='0.3',
      description='Sequential model-based optimization toolbox.',
      long_description=('Scikit-Optimize, or `skopt`, is a simple and efficient'
                        ' library to minimize (very) expensive and noisy'
                        ' black-box functions. It implements several methods'
                        ' for sequential model-based optimization.'),
      url='https://scikit-optimize.github.io/',
      license='BSD',
      author='The scikit-optimize contributors',
      packages=['skopt', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process'],
      install_requires=["numpy", "scipy", "scikit-learn>=0.18",
                        "scikit-garden", "matplotlib"]
      )
