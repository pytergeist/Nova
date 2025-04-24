#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"

rm -rf build
mkdir build
cd build

cmake -DPython_EXECUTABLE="$(which python)" ..
make
