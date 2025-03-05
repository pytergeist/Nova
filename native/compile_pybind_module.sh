c++ -std=c++20 -arch arm64 -shared -undefined dynamic_lookup \
  -I/opt/homebrew/include $(python3-config --includes) \
  tensor_ops.cpp -o vector_math.so
