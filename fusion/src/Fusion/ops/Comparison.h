#ifndef OPS_COMPARISON_H
#define OPS_COMPARISON_H

#include <vector>
#include <string_view>
#include <vector>
#include "../Tensor.h"
#include "../core/ElementWise.h"
#include "Helpers.h"


namespace math {
    template <typename T>
    inline Tensor<T> greater(const Tensor<T> &x, const Tensor<T> &y) {
        std::vector<size_t> out_shape;
        std::vector<T> out_data;
        ewise::binary_ewise_tag<T, GreaterThanSIMD>(x, y, out_shape,
                                                    out_data);
        return Tensor(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
    }

    template <typename T>
    inline Tensor<T> greater_equal(const Tensor<T> &x, const Tensor<T> &y) {
        std::vector<size_t> out_shape;
        std::vector<T> out_data;
        ewise::binary_ewise_tag<T, GreaterThanEqualSIMD>(x, y, out_shape,
                                                         out_data);
        return Tensor(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
    }

    template <typename T>
    inline Tensor<T> maximum(const Tensor<T> &x, const Tensor<T> &y) {
        std::vector<size_t> out_shape;
        std::vector<T> out_data;
        ewise::binary_ewise_tag<T, MaximumSIMD>(x, y, out_shape, out_data);
        return Tensor(std::move(out_shape), std::move(out_data), Device::CPU, grad_flow(x, y));
    }
}


#endif // OPS_COMPARISON_H
