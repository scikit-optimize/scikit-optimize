#!/bin/bash
# Almost copied verbatim from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/circle/push_doc.sh

if [ -z $CIRCLE_PROJECT_USERNAME ];
then USERNAME="skoptci";
else USERNAME=$CIRCLE_PROJECT_USERNAME;
fi

MSG="Pushing the docs for revision for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

# Copying to github pages
echo "Copying built files"
git clone -b master "git@github.com:scikit-optimize/scikit-optimize.github.io" deploy
cd deploy
git rm -r notebooks/*
git rm -r rtd/*
git rm -r beta/*
cd ..
cp -r ${HOME}/doc/skopt/* deploy
# Move into deployment directory
cd deploy

# Commit changes, allowing empty changes (when unchanged)
echo "Committing and pushing to Github"
echo "$USERNAME"
git config --global user.name $USERNAME
git config --global user.email "skoptci@gmail.com"
git config --global push.default matching
git add -A
git commit --allow-empty -m "$MSG"
git push

echo "$MSG"
