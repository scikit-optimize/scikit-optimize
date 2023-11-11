#!/bin/bash
# Adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/circle/push_doc.sh
# The scikit-learn and scikit-optimize developers.
# License: BSD-style
#
# This script is meant to be called in the "deploy" step.

set -eux

DOC_REPO="scikit-optimize.github.io"
DOC_REPO_URL="https://:$GITHUB_TOKEN@github.com/scikit-optimize/$DOC_REPO"
GENERATED_DOC_DIR=$1
BRANCH="${GITHUB_REF#refs/*/}"
BRANCH="${BRANCH:-master}"

if [[ -z "$GENERATED_DOC_DIR" ]]; then
    echo "Need to pass directory of the generated doc as argument"
    echo "Usage: $0 <generated_doc_dir>"
    exit 1
fi

# Absolute path needed because we use cd further down in this script
GENERATED_DOC_DIR=$(readlink -f $GENERATED_DOC_DIR)

if [ "$BRANCH" = "master" ]
then
    dir=dev
else
    # Strip off .X
    dir="${BRANCH%.*}"
fi

MSG="Pushing docs to $dir/ for branch: $BRANCH, commit $GITHUB_SHA"

if [ ! -d $DOC_REPO ];
then git clone "$DOC_REPO_URL"
fi
cd $DOC_REPO

# check if it's a new branch

if [ -d $dir ]
then
	git rm -rf $dir/
fi
cp -R $GENERATED_DOC_DIR $dir
git config user.email "skoptci@gmail.com"
git config user.name "skoptci"
git config push.default matching
git add -f $dir/
git commit -m "$MSG" $dir
git push
echo $MSG
