# Adapted shamelessly from https://github.com/scikit-learn-contrib/project-template/blob/master/ci_scripts/install.sh

echo_requirements_string() {
    # Echo a requirement string for example
    # "pip pytest python='2.7.3 scikit-learn=*". It has a hardcoded
    # list of possible packages to install and looks at _VERSION
    # environment variables to know whether to install a given package and
    # if yes which version to install. For example:
    #   - for numpy, NUMPY_VERSION is used
    #   - for scikit-learn, SCIKIT_LEARN_VERSION is used
    TO_INSTALL_ALWAYS="pip pytest nose"
    REQUIREMENTS="$TO_INSTALL_ALWAYS"
    TO_INSTALL_MAYBE="numpy scipy matplotlib scikit-learn pyaml joblib"
    for PACKAGE in $TO_INSTALL_MAYBE; do
        # Capitalize package name and add _VERSION
        PACKAGE_VERSION_VARNAME="${PACKAGE^^}_VERSION"
        # replace - by _, needed for scikit-learn for example
        PACKAGE_VERSION_VARNAME="${PACKAGE_VERSION_VARNAME//-/_}"
        # dereference $PACKAGE_VERSION_VARNAME to figure out the
        # version to install
        PACKAGE_VERSION="${!PACKAGE_VERSION_VARNAME}"
        if [[ -n "$PACKAGE_VERSION" ]]; then
            if [[ "$PACKAGE_VERSION" == "*" ]]; then
                REQUIREMENTS="$REQUIREMENTS $PACKAGE"
            else
                REQUIREMENTS="$REQUIREMENTS $PACKAGE==$PACKAGE_VERSION"
            fi
        fi
    done
    echo $REQUIREMENTS
}

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
   then
   wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O miniconda.sh
   fi
chmod +x miniconda.sh && ./miniconda.sh -b
cd ..
export PATH="$HOME/miniconda3/bin:$PATH"
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip nose pytest
source activate testenv
REQUIREMENTS=$(echo_requirements_string)
pip install $PIP_FLAGS ${REQUIREMENTS}
if [[ "$COVERAGE" == "true" ]]; then
    pip install pytest-cov coverage coveralls
fi

if [[ "$SDIST" == "true" ]]; then
    python setup.py sdist
    pip install twine
else
    pip install -e '.[plots]'
fi
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
