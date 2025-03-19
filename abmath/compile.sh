#!/bin/bash
rm -rf build
mkdir build
cd build || exit 1
cmake -DPYTHON_EXECUTABLE:FILEPATH="$(which python)" ..
make
