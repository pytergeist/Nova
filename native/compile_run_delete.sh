c++ -std=c++20 -arch arm64 -I/opt/homebrew/include $(python3-config --includes) tensor_ops.cpp -o main $(python3-config --ldflags)
./main
rm main
