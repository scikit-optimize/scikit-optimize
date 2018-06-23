try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import versioneer


setup(name='scikit-optimize',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Sequential model-based optimization toolbox.',
      long_description=open('README.rst').read(),
      url='https://scikit-optimize.github.io/',
      license='BSD',
      author='The scikit-optimize contributors',
      packages=['skopt', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process'],
      install_requires=['pyaml', 'numpy', 'scipy>=0.14.0',
                        'scikit-learn>=0.19.1'],
      extras_require={
        'plots':  ["matplotlib"]
        }
      )
