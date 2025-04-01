#!/bin/bash
# This script will find all C/C++ source and header and header impl files
# and format them in-place using clang-format.

find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -exec clang-format -i {} +