c++ -std=c++20 -arch arm64 -shared -undefined dynamic_lookup \
  -I/opt/homebrew/include $(python3-config --includes) \
  src/AbTensor/core/MathOps.cpp -o binaries/abtensor.so
