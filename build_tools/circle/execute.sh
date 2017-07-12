export PATH="$HOME/miniconda3/bin:$PATH"
source activate testenv
export SKOPT_HOME=$(pwd)

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

cd ${SKOPT_HOME}/docs
make html
cp -r ./_build/html ${CIRCLE_ARTIFACTS}
