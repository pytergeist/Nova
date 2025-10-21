#ifndef OPS_MATMUL_H
#define OPS_MATMUL_H

#include <vector>
#include <string_view>
#include <vector>
#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "../kernels/Blas.h"
#include "Helpers.h"



namespace math { namespace linalg {
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
        return Tensor<T>(std::move(out_shape), std::move(data), Device::CPU, grad_flow(x, y));
    }

   template <typename T>
   inline Tensor<T> swapaxes(const Tensor<T> &x, int axis1, int axis2) {
        std::vector<size_t> out_shape = x.shape_;
        axis1 = serial::normalise_axis(axis1, x.rank());
        axis2 = serial::normalise_axis(axis2, x.rank());
        std::swap(out_shape[axis1], out_shape[axis2]);
        std::vector<T> out =
            serial::swapaxes<T>(x, x.shape_, axis1, axis2);
        return Tensor<T>(std::move(out_shape), std::move(out), Device::CPU);
   }
}
}


#endif // OPS_MATMUL_H
