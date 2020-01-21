source activate testenv
export SKOPT_HOME=$(pwd)

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# Generating documentation
for nb in examples/*ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=3600 --execute "$nb" --to markdown |& tee nb_to_md.txt
    traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt)
    if [[ $traceback ]]; then
        exit 1
    fi
done

cd ~
mkdir -p ${HOME}/doc/skopt/notebooks
cp ${SKOPT_HOME}/examples/*md ${HOME}/doc/skopt/notebooks
find ${SKOPT_HOME}/examples -name \*_files -exec cp -r {} ${HOME}/doc/skopt/notebooks \;
python ${SKOPT_HOME}/build_tools/circle/make_doc.py --overwrite --html --html-dir ./doc --template-dir ${SKOPT_HOME}/build_tools/circle/templates --notebook-dir ./doc/skopt/notebooks skopt
cd ${SKOPT_HOME}/docs && sphinx-build -M html ${HOME}/doc/skopt/docs/source ${HOME}/doc/skopt/rtd -W --keep-going
