c++ main.cpp \
  -std=c++20 -O3 -mcpu=apple-m1 \
  -I../../build/_deps/xsimd-src/include \
  -I. \
  -I/opt/homebrew/include/eigen3 \
  -I/opt/homebrew/opt/openblas/include \
  -L/opt/homebrew/opt/openblas/lib -lopenblas \
  -o main

./main
