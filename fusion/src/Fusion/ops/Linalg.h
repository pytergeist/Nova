#ifndef OPS_MATMUL_H
#define OPS_MATMUL_H

#include <vector>
#include <string_view>
#include <vector>
#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "../kernels/Blas.h"



namespace ops {
    template <typename T>
    inline Tensor<T> matmul(const Tensor<T> &x, const Tensor<T> &y) {
        auto const &shapeA = x.shape_;
        auto const &shapeB = y.shape_;
        size_t rank = shapeA.size();
        size_t m = shapeA[rank - 2];
        size_t n = shapeB[rank - 1];
        std::vector<size_t> out_shape = shapeA;
        out_shape[rank - 1] = n;
        size_t batch = 1;
        for (size_t i = 0; i < rank - 2; ++i) {
            batch *= shapeA[i];
        }
        std::vector<T> data(batch * m * n);
        blas_ops::matmul<T>(x, shapeA, y, shapeB, data);
        return Tensor<T>(std::move(out_shape), std::move(data), Device::CPU);
    }

}


#endif // OPS_MATMUL_H
