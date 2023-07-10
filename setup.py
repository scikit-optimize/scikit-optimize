import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    license = fh.read()

import scikit-optimize.skopt as skopt

setuptools.setup(
    name='scikit-optimize',
    version='0.1',
    license=license,
    packages=['skopt', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process', 'skopt.sampler'],
    #packages=setuptools.find_packages(
    #    exclude=('doc')
    #),
    url='https://github.com/MikeSmithLab/scikit-optimize',
    
    install_requires=['joblib>=0.11', 'pyaml>=16.9', 'numpy>=1.13.3',
                        'scipy>=0.19.1',
                        'scikit-learn>=0.20.0'],   
    include_package_data=True,
)
