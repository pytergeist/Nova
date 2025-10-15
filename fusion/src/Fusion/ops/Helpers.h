#ifndef OP_HELPERS_H
#define OP_HELPERS_H

#include "../Tensor.h"

template <typename T>
bool grad_flow(const Tensor<T>& x, const Tensor<T>& y) {
    return x.requires_grad() || y.requires_grad();
};

#endif // OP_HELPERS_H
