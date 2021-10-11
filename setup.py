from setuptools import find_packages, setup


if __name__ == '__main__':
    setup(
        name='scikit-optimize',
        description='Sequential model-based optimization toolbox.',
        long_description=open('README.rst').read(),
        url='https://scikit-optimize.github.io/',
        project_urls={
            'Documentation': 'https://scikit-optimize.github.io/',
            'Source': 'https://github.com/scikit-optimize/scikit-optimize',
            'Tracker': 'https://github.com/scikit-optimize/scikit-optimize/issues',
        },
        license='BSD-3-Clause',
        author='The scikit-optimize contributors',
        packages=find_packages(include=('skopt.*',),
                               exclude=('*.tests',)),
        use_scm_version=True,
        python_requires='>= 3.6',
        setup_requires=[
            'setuptools_scm',
        ],
        install_requires=[
            'joblib>=0.11',
            'pyaml>=16.9',
            'numpy>=1.13.3',
            'scipy>=0.19.1',
            'scikit-learn>=0.20.0',
            'importlib-metadata; python_version < "3.8"'
        ],
        extras_require={
            'plots': [
                "matplotlib>=2.0.0",
            ],
            'dev': [
                'flake8',
                'pytest',
                'pytest-cov',
                'pytest-xdist',
            ],
            'doc': [
                'sphinx',
                'sphinx-gallery>=0.6',
                'memory_profiler',
                'numpydoc',
            ],
        },
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Visualization',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: OS Independent',
            'Operating System :: POSIX',
            'Operating System :: Unix',
        ],
    )
