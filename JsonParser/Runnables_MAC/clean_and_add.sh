#!/bin/bash
set -e

if [ "$1" == "--fresh" ]; then
    ./clean_inputs.sh --fresh
fi

./clean_inputs.sh
./add_vectors.sh