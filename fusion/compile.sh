#!/usr/bin/env bash
set -e

rm -rf build
mkdir build
cd build

cmake \
  -DPython_EXECUTABLE:FILEPATH="$(which python)" \
  -DPython_EXECUTABLE:FILEPATH="$(which python)" \
  -DCMAKE_PREFIX_PATH="/opt/homebrew;/opt/homebrew/Cellar/spack/0.23.1/.../xsimd-8.1.0-.../include" \
  -DGTest_DIR="/opt/homebrew/Cellar/googletest/1.11.0/lib/cmake/GTest" \
  ..

make
