#!/bin/bash

set -e
set -x
if [[ "$SDIST" != "true" ]]; then
    pushd .
    cd doc
    make doctest
    popd
fi
