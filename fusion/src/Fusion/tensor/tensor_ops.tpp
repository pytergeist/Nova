#pragma once
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cblas.h>

/**
 * @brief Helper function that applies a binary operation element-wise.
 *
 * This function checks for three conditions:
 *   - If both tensors have the same size, it applies op to each pair of
 * corresponding elements.
 *   - If the left tensor is a scalar (size 1), it applies op between its single
 * value and each element of the right tensor.
 *   - If the right tensor is a scalar (size 1), it applies op between each
 * element of the left tensor and that single value. Otherwise, it throws an
 * exception.
 *
 * @tparam T The element type.
 * @tparam BinaryOp The binary operation type.
 * @param a Left tensor.
 * @param b Right tensor.
 * @param op Binary operation to perform.
 * @return Tensor<T> The resulting tensor.
 */

template<typename T, typename UnaryOp>
Tensor<T> elementwise_unary_op(const Tensor<T> &a, UnaryOp op) {
    std::vector<T> result;
    if (a.arr.size() > 1) {
        // TODO: This won't work for dim > 2??
        result.resize(a.arr.size());
        for (size_t i = 0; i < a.arr.size(); i++) {
            result[i] = op(a.arr[i]);
        }
    } else if (a.arr.size() == 1) {
        result.resize(a.arr.size());
        result[0] = op(a.arr[0]);
    } else {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    return Tensor<T>(result, a.shape);
}

template<typename T, typename BinaryOp>
Tensor<T> elementwise_binary_op(const Tensor<T> &a, const Tensor<T> &b,
                                BinaryOp op) {
    std::vector<T> result;
    if (a.arr.size() == b.arr.size()) {
        result.resize(a.arr.size());
        for (size_t i = 0; i < a.arr.size(); ++i)
            result[i] = op(a.arr[i], b.arr[i]);
        return Tensor<T>(result, a.shape);
    }
    if (a.arr.size() == 1) {
        result.resize(b.arr.size());
        for (size_t i = 0; i < b.arr.size(); ++i)
            result[i] = op(a.arr[0], b.arr[i]);
        return Tensor<T>(result, b.shape);
    }
    if (b.arr.size() == 1) {
        result.resize(a.arr.size());
        for (size_t i = 0; i < a.arr.size(); ++i)
            result[i] = op(a.arr[i], b.arr[0]);
        return Tensor<T>(result, a.shape);
    }
    throw std::invalid_argument("Tensor sizes do not match");
}

// Constructor definitions

// Constructor from a vector.
template<typename T> // TODO: Add error checking for mismatched shape and data
// size
Tensor<T>::Tensor(const std::vector<T> &data, const std::vector<size_t> &shape)
    : arr(data), shape(shape) {
    size_t expected = 1;
    for (auto const dim: shape) {
        expected *= dim;
    }
    if (expected != arr.size()) {
        throw std::invalid_argument(
            "The provided Tensor shape does not match data shape");
    }
}

// Constructor from a scalar.
template<typename T>
Tensor<T>::Tensor(const T &value) : arr(1, value), shape({1}) {
}

/* Operator overloads for binary operations (all now work with Tensor<T>
 * arguments) and take one argument (other tensor), the binary operations are
 * applied elementwise between *this (current tensor and other tensor).
 */

// tensor + tensor
template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &tensor) const {
    return elementwise_binary_op(*this, tensor, std::plus<T>());
}

// tensor - tensor
template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &tensor) const {
    return elementwise_binary_op(*this, tensor, std::minus<T>());
}

// tensor * tensor
template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &tensor) const {
    return elementwise_binary_op(*this, tensor, std::multiplies<T>());
}

// tensor / tensor
template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &tensor) const {
    return elementwise_binary_op(*this, tensor, std::divides<T>());
}

// tensor (pow) tensor
template<typename T>
Tensor<T> Tensor<T>::pow(const Tensor<T> &tensor) const {
    return elementwise_binary_op(
        *this, tensor, [](T base, T exp) -> T { return std::pow(base, exp); });
}

/* Operator overloads for unary operations.
 * Takes no arguments (), the binary operations are applied elementwise
 * to *this (current tensor).
 */

// sqrt(tensor)
template<typename T>
Tensor<T> Tensor<T>::sqrt() const {
    return elementwise_unary_op(*this,
                                [](T base) -> T { return std::sqrt(base); });
}

// exp(tensor)
template<typename T>
Tensor<T> Tensor<T>::exp() const {
    return elementwise_unary_op(*this,
                                [](T base) -> T { return std::exp(base); });
}

// log(tensor)
template<typename T>
Tensor<T> Tensor<T>::log() const {
    return elementwise_unary_op(*this,
                                [](T base) -> T { return std::log(base); });
}

// sum(tensor)
template<typename T>
Tensor<T> Tensor<T>::sum() const {
    T total = std::accumulate(arr.begin(), arr.end(), T());
    return Tensor<T>(total);
}

// matmul fn for 2D Tensor @ 2D Tensor
template<typename T>
Tensor<T> matrix_2d_op(const Tensor<T> &tensor1, const Tensor<T> &tensor2) {

    const size_t m = tensor1.shape[0];
    const size_t n = tensor1.shape[1];
    const size_t p = tensor2.shape[1];
    const size_t result_size = m * p;
    std::vector<double> result(result_size, 0.0);

    const double alpha = 1.0;
    const double beta  = 0.0;


    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, p, n,
                alpha,
                tensor1.arr.data(), n,
                tensor2.arr.data(), p,
                beta,
                result.data(), p);
    return Tensor<T>(result, {m, p});
}

// matmul fn for 2D Tensor @ 1D Tensor
template<typename T>
Tensor<T> matrix_1d_op(const Tensor<T> &tensor1, const Tensor<T> &tensor2) {
    const size_t m = tensor1.shape[0];
    const size_t n = tensor1.shape[1];
    std::vector<T> result(m, 0.0);

    const double alpha = 1.0;
    const double beta  = 0.0;

    cblas_dgemv(CblasRowMajor,
                CblasNoTrans,
                m,
                n,
                alpha,
                tensor1.arr.data(),
                n,
                tensor2.arr.data(),
                1,
                beta,
                result.data(),
                1);
    return Tensor<T>(result, {m});
}

// matmul(tensor)
template<typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T> &tensor) const {
    if (this->shape.size() == 2 && tensor.shape.size() == 2 &&
        // TODO: remove hard coding and abstract checks into diff fns
        this->shape[1] == tensor.shape[0]) {
        return matrix_2d_op(*this, tensor);
    }
    if (this->shape.size() == 2 && tensor.shape.size() == 1 &&
        this->shape[1] == tensor.shape[0]) {
        return matrix_1d_op(*this, tensor);
    }
    throw std::invalid_argument("Tensor sizes do not match");
}

template<typename T>
Tensor<T> Tensor<T>::transpose() const {
    // Ensure the tensor is 2D.
    if (this->shape.size() != 2) {
        throw std::invalid_argument("Transpose only supports 2D tensors");
    }

    size_t m = this->shape[0];
    size_t n = this->shape[1];

    std::vector<T> transposed(m * n, T{});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            transposed[j * m + i] = this->arr[i * n + j];
        }
    }

    return Tensor<T>(transposed, {n, m});
}

template<typename T>
Tensor<T> Tensor<T>::maximum(const Tensor<T> &tensor) const {
    // 2D tensors
    if (this->shape.size() == 2) {
        size_t const m = this->shape[0];
        size_t const n = this->shape[1];
        std::vector<T> result(m * n, T{});
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                if (this->arr[i * n + j] > tensor.arr[0]) {
                    result[i * n + j] = this->arr[i * n + j];
                } else {
                    result[i * n + j] = tensor.arr[0];
                }
            }
        }
        return Tensor<T>(result, {m, n});
    }
    if (this->shape.size() == 1) {
        size_t const m = this->shape[0];
        std::vector<T> result(m, T{});
        for (size_t i = 0; i < m; i++) {
            if (this->arr[i] > tensor.arr[0]) {
                result[i] = this->arr[i];
            }
            else {
                result[i] = tensor.arr[0];
            }
        }
        return Tensor<T>(result, {m});
    }
    throw std::invalid_argument("Tensor sizes do not match");
}
