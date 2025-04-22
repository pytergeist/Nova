#ifndef TENSOR_ARITHMETIC_IPP
#define TENSOR_ARITHMETIC_IPP

#include "eigen_helpers.h"

// +, -, *, /:
template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    return tensor_detail::binary_elementwise_op(
        *this, other,
        [](auto& A, auto& B){ return A + B; },
        "addition"
    );
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
    return tensor_detail::binary_elementwise_op(
        *this, other,
        [](auto& A, auto& B){ return A - B; },
        "subtraction"
    );
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
    return tensor_detail::binary_elementwise_op(
        *this, other,
        [](auto& A, auto& B){ return (A.array() * B.array()).matrix(); },
        "multiplication"
    );
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
    return tensor_detail::binary_elementwise_op(
        *this, other,
        [](auto& A, auto& B){ return (A.array() / B.array()).matrix(); },
        "division"
    );
}

// Unary ops:
template<typename T>
Tensor<T> Tensor<T>::sqrt() const {
    return tensor_detail::unary_elementwise_op(
        *this,
        [](auto& A){ return A.array().sqrt().matrix();},
        "sqrt"
    );
}

template<typename T>
Tensor<T> Tensor<T>::exp() const {
    return tensor_detail::unary_elementwise_op(
        *this,
        [](auto& A){ return A.array().exp().matrix(); },
        "exp"
    );
}

template<typename T>
Tensor<T> Tensor<T>::log() const {
    return tensor_detail::unary_elementwise_op(
        *this,
        [](auto& A){ return A.array().log().matrix(); },
        "log"
    );
}

template<typename T>
Tensor<T> Tensor<T>::pow(const T exponent) const {
    return tensor_detail::unary_elementwise_op(
        *this,
        [exponent](auto& A){ return A.array().pow(exponent).matrix(); },
        "pow(scalar)"
    );
}

template<typename T>
Tensor<T> Tensor<T>::pow(const Tensor<T>& exponent) const {
    return tensor_detail::binary_elementwise_op(
        *this, exponent,
        [](auto& A, auto& B){
            return A.binaryExpr(B, [](auto a, auto b){ return std::pow(a,b); });
        },
        "pow(tensor)"
    );
}

#endif // TENSOR_ARITHMETIC_IPP
