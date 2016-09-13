# Adapted shamelessly from https://github.com/scikit-learn-contrib/project-template/blob/master/ci_scripts/install.sh
# Deactivate the circleci-provided virtual environment and setup a
# conda-based environment instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Use the miniconda installer for faster download / install of conda
# itself
# XXX: Most of this is very similar to travis/install.sh. We should
# probably refactor it later.
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
conda create -n testenv --yes python pip nose \
   numpy scipy cython matplotlib
source activate testenv
pip install git+http://github.com/scikit-learn/scikit-learn.git

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
export SKOPT_HOME=$(pwd)

conda install --yes jupyter
pip install pdoc==0.3.2 pygments

# Generating documentation
for nb in examples/*ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=900 --execute "$nb" --to markdown |& tee -a nb_to_md.txt
done

cd ~
mkdir -p ./doc/skopt/notebooks
cp ${SKOPT_HOME}/examples/*md ${HOME}/doc/skopt/notebooks
cp -r ${SKOPT_HOME}/examples/*_files ${HOME}/doc/skopt/notebooks
python ${SKOPT_HOME}/build_tools/circle/make_doc.py --overwrite --html --html-dir ./doc --template-dir ${SKOPT_HOME}/build_tools/circle/templates --notebook-dir ./doc/skopt/notebooks skopt
cp -r ./doc ${CIRCLE_ARTIFACTS}
