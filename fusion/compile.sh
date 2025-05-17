#!/usr/bin/env bash
set -e

rm -rf build
mkdir build
cd build

cmake \
  -DPython_EXECUTABLE:FILEPATH="$(which python)" \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/Cellar/spack/0.23.1/opt/spack/darwin-sequoia-m1/apple-clang-17.0.0/xsimd-8.1.0-s3yjkaw2c7iwa6wpqkcofp5fgzk6xhsw" \
  ..

make
