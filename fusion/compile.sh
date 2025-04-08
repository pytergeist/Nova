#!/bin/bash
rm -rf build
mkdir build
cd build || exit
cmake -DPython_EXECUTABLE:FILEPATH="$(which python)" ..
cmake -DCMAKE_PREFIX_PATH="/opt/homebrew" ..
make
