#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_file> <output_file>"
    exit 1
fi

source_file="$1"
output_file="$2"

c++ -std=c++20 -arch arm64 -I/opt/homebrew/include -Isrc/templates $(python3-config --includes) "$source_file" -o "$output_file" $(python3-config --ldflags)
