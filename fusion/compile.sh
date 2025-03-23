#!/bin/bash
rm -rf build
mkdir build
cd build
cmake -DPython_EXECUTABLE:FILEPATH="$(which python)" ..
make
