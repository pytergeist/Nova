#ifndef TENSOR_H
#define TENSOR_H
#include "kernels/blas_api.cpp"
#include "kernels/serial_api.cpp"
#include "kernels/xsimd_api.cpp"
#include "storage/dense_storage.h"
#include "storage/storage_interface.h"
#include "xsimd/xsimd.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

template<typename T>
class Tensor {
public:
    std::unique_ptr<ITensorStorage<T> > storage;
    std::vector<size_t> shape_;
    size_t rank_;

    explicit Tensor(std::vector<size_t> shape, std::vector<T> data,
                    Device device = Device::CPU)
        : shape_(std::move(shape)) {
        if (device == Device::CPU) {
            // Use the stored shape_ (guaranteed non-empty if you passed one in)
            storage = std::make_unique<NDTensorStorage<T> >(shape_, std::move(data));
            rank_ = storage->ndims();
        } else {
            throw std::invalid_argument("Unsupported device type");
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        const auto *cpuStorage =
                dynamic_cast<const NDTensorStorage<T> *>(tensor.storage.get());
        if (cpuStorage) {
            const std::vector<T> &data = cpuStorage->data();
            size_t size = cpuStorage->size();
            os << "Tensor(";
            for (size_t i = 0; i < size; i++) {
                os << data[i];
                if (i + 1 < size)
                    os << ", ";
            }
            os << ")" << std::endl;
        } else {
            os << "Tensor(unsupported storage type)";
        }
        return os;
    }

    std::vector<T> &raw_data() { return storage->data(); }
    const std::vector<T> &raw_data() const { return storage->data(); }
    [[nodiscard]] size_t flat_size() const { return storage->size(); }

    Tensor<T> operator+(const Tensor<T> &other) const {
        // 1) decide on broadcast shape & length
        std::vector<size_t> out_shape =
                (other.flat_size() == 1 ? this->shape_ : other.shape_);
        std::size_t out_size =
                (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

        const T *a_ptr = this->storage->data_ptr();
        const T *b_ptr = other.storage->data_ptr();
        std::size_t na = this->flat_size();
        std::size_t nb = other.flat_size();

        std::vector<T> result(out_size);
        T *r_ptr = result.data();

        using arch = xsimd::default_arch;
        using tag = xsimd::unaligned_mode;
        xsimd_ops::add{}(arch{}, a_ptr, na, b_ptr, nb, r_ptr, out_size, tag{});

        return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
    }

    Tensor<T> operator-(const Tensor<T> &other) const {
        std::vector<size_t> out_shape =
                (other.flat_size() == 1 ? this->shape_ : other.shape_);
        std::size_t out_size =
                (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

        const T *a_ptr = this->storage->data_ptr();
        const T *b_ptr = other.storage->data_ptr();
        std::size_t na = this->flat_size();
        std::size_t nb = other.flat_size();

        std::vector<T> result(out_size);
        T *r_ptr = result.data();
        xsimd_ops::subtract{}(xsimd::default_arch{}, a_ptr, na, b_ptr, nb, r_ptr,
                              out_size, xsimd::aligned_mode{});

        return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
    }

    Tensor<T> &operator-=(const Tensor<T> &other) {
        const size_t out_size =
                (other.flat_size() == 1 ? flat_size() : other.flat_size());
        const size_t na = flat_size();
        const size_t nb = other.flat_size();
        T *self_ptr = storage->data_ptr();
        const T *other_ptr = other.storage->data_ptr();

        xsimd_ops::subtract{}(xsimd::default_arch{}, self_ptr, na, other_ptr, nb,
                              self_ptr, out_size, xsimd::unaligned_mode{});

        return *this;
    }

    Tensor<T> operator/(const Tensor<T> &other) const {
        std::vector<size_t> out_shape =
                (other.flat_size() == 1 ? this->shape_ : other.shape_);
        std::size_t out_size =
                (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

        const T *a_ptr = this->storage->data_ptr();
        const T *b_ptr = other.storage->data_ptr();
        std::size_t na = this->flat_size();
        std::size_t nb = other.flat_size();

        std::vector<T> result(out_size);
        T *r_ptr = result.data();

        xsimd_ops::divide{}(xsimd::default_arch{}, a_ptr, na, b_ptr, nb, r_ptr,
                            out_size, xsimd::aligned_mode{});

        return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
    }

    //     // overload the + operator
    Tensor<T> operator*(const Tensor<T> &other) const {
        std::vector<size_t> out_shape =
                (other.flat_size() == 1 ? this->shape_ : other.shape_);
        std::size_t out_size =
                (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

        const T *a_ptr = this->storage->data_ptr();
        const T *b_ptr = other.storage->data_ptr();
        std::size_t na = this->flat_size();
        std::size_t nb = other.flat_size();

        std::vector<T> result(out_size);
        T *r_ptr = result.data();

        xsimd_ops::multiply{}(xsimd::default_arch{}, a_ptr, na, b_ptr, nb, r_ptr,
                              out_size, xsimd::aligned_mode{});

        return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
    }

    Tensor<T> operator>=(const Tensor<T> &other) const {
        std::vector<size_t> out_shape =
                (other.flat_size() == 1 ? this->shape_ : other.shape_);
        std::size_t out_size =
                (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

        const T *a_ptr = this->storage->data_ptr();
        const T *b_ptr = other.storage->data_ptr();
        std::size_t na = this->flat_size();
        std::size_t nb = other.flat_size();

        std::vector<T> result(out_size);
        T *r_ptr = result.data();

        xsimd_ops::greater_than_equal_to_numeric{}(xsimd::default_arch{}, a_ptr, na, b_ptr, nb, r_ptr,
                                                out_size, xsimd::aligned_mode{});

        return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
    }

    Tensor<T> maximum(const Tensor<T> &other) const {
        std::vector<size_t> out_shape =
                (other.flat_size() == 1 ? this->shape_ : other.shape_);
        std::size_t out_size =
                (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

        const T *a_ptr = this->storage->data_ptr();
        const T *b_ptr = other.storage->data_ptr();
        std::size_t na = this->flat_size();
        std::size_t nb = other.flat_size();

        std::vector<T> result(out_size);
        T *r_ptr = result.data();

        xsimd_ops::maximum{}(xsimd::default_arch{}, a_ptr, na, b_ptr, nb, r_ptr,
                             out_size, xsimd::aligned_mode{});

        return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
    }

    Tensor<T> sqrt() {
        std::vector<size_t> shape = this->shape_;
        size_t size = this->flat_size();
        std::vector<T> data;
        data.resize(size);
        std::vector<T> v1 = this->raw_data();
        using arch = xsimd::default_arch;
        using tag = xsimd::unaligned_mode;
        xsimd_ops::sqrt{}(arch{}, v1, data, tag{});
        return Tensor<T>(shape, data, Device::CPU);
    };

    Tensor<T> exp() {
        std::vector<size_t> shape = this->shape_;
        size_t size = this->flat_size();
        std::vector<T> data;
        data.resize(size);
        std::vector<T> v1 = this->raw_data();
        using arch = xsimd::default_arch;
        using tag = xsimd::unaligned_mode;
        xsimd_ops::exp{}(arch{}, v1, data, tag{});
        return Tensor<T>(shape, data, Device::CPU);
    };

    //
    Tensor<T> log() {
        std::vector<size_t> shape = this->shape_;
        size_t size = this->flat_size();
        std::vector<T> data;
        data.resize(size);
        std::vector<T> v1 = this->raw_data();
        using arch = xsimd::default_arch;
        using tag = xsimd::unaligned_mode;
        xsimd_ops::log{}(arch{}, v1, data, tag{});
        return Tensor<T>(shape, data, Device::CPU);
    };

    Tensor<T> pow(Tensor<T> &other) const {
        std::vector<size_t> out_shape =
                (other.flat_size() == 1 ? this->shape_ : other.shape_);
        std::size_t out_size =
                (other.flat_size() == 1 ? this->flat_size() : other.flat_size());

        const T *a_ptr = this->storage->data_ptr();
        const T *b_ptr = other.storage->data_ptr();
        std::size_t na = this->flat_size();
        std::size_t nb = other.flat_size();

        std::vector<T> result(out_size);
        T *r_ptr = result.data();

        xsimd_ops::pow{}(xsimd::default_arch{}, a_ptr, na, b_ptr, nb, r_ptr,
                         out_size, xsimd::aligned_mode{});

        return Tensor<T>(std::move(out_shape), std::move(result), Device::CPU);
    }

    //
    Tensor<T> sum() {
        std::vector<T> data(1);
        std::vector<T> &in = raw_data();

        using arch = xsimd::default_arch;
        xsimd_ops::sum{}(arch{}, in, data);

        return Tensor<T>({1}, std::move(data), Device::CPU);
    }

    //
    Tensor<T> matmul(Tensor<T> &other) {
        auto const &shapeA = this->shape_;
        auto const &shapeB = other.shape_;
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

        blas_ops::matmul(this->raw_data(), shapeA, other.raw_data(), shapeB, data);

        return Tensor<T>(std::move(out_shape), std::move(data), Device::CPU);
    }

    //

    Tensor<T> transpose() {
        std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());

        size_t size = flat_size();
        std::vector<T> new_data(size);

        serial_ops::transpose(this->raw_data(), this->shape_, new_data);

        return Tensor<T>(std::move(new_shape), std::move(new_data), Device::CPU);
    }
};

#endif // TENSOR_H
