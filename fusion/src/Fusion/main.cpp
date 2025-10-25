#include <iostream>
#include <vector>
#include <cstddef>
#include <vector>

#include "autodiff/Engine.h"
#include "autodiff/Graph.h"
#include "autodiff/Node.h"
#include "autodiff/NodeInterface.h"
#include "autodiff/Sort.h"
#include "autodiff/policies/LinAlg/LinAlg.h"
#include "autodiff/policies/Transcendental/Transcendental.h"
#include "autodiff/policies/Reduction/Reduction.h"
#include "autodiff/policies/Reduction/Sum.h"
#include "autodiff/policies/Operation.h"
#include "Tensor.h"
#include "storage/TensorBuffer.h"
#include "autodiff/policies/Comparison/Comparison.h"
#include "autodiff/policies/Shape/Shape.h"
#include "autodiff/policies/Ewise/Ewise.h"
#include "autodiff/EngineContext.h"

#include <vector>
#include <random>
#include <cstddef>



std::vector<float> rand_matrix_flat(size_t rows, size_t cols, uint32_t seed=123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> buf;
    buf.reserve(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        buf.push_back(dist(rng));
    }
    return buf; // size == rows*cols; row-major implicit
}

int main() {
    using T = float;
    using AddOp = Operation<T, Add<T>>;
    using ExpOp = Operation<T, Exp<T>>;
    using divOp = Operation<T, Divide<T>>;
    using MulOp = Operation<T, Multiply<T>>;
    using subOp = Operation<T, Subtract<T>>;
    using logOp = Operation<T, Log<T>>;
    using sqrtOp = Operation<T, Sqrt<T>>;
    using powOp = Operation<T, Pow<T>>;
    using sumOp = Operation<T, Sum<T>>;
    using maximumOp = Operation<T, Maximum<T>>;
    using greaterthanOp = Operation<T, GreaterThan<T>>;
    using transposeOp = Operation<T, Transpose<T>>;
    using matmulOp = Operation<T, MatMul<T>>;

    auto a_data = rand_matrix_flat(10, 10, /*seed=*/42);
    auto b_data = rand_matrix_flat(10, 10, /*seed=*/43);

    Tensor<float> A({10,10}, a_data, Device::CPU, /*requires_grad=*/true);
    Tensor<float> B({10,10}, b_data, Device::CPU, /*requires_grad=*/true);
    Engine<T> engine;
    EngineContext<T>::set(&engine);

    Tensor<T> C = A - B;
    Tensor<T> D = C * B;
    Tensor<T> E = C / D;
    Tensor<T> F = A * E;

    std::cout << C << std::endl;

    D.backward();
    std::cout << F.grad() << std::endl;






}
