#include "tensor.h"


Tensor Tensor::operator+(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] + tensor.arr[i];
    }
    return Tensor(new_arr);
}


Tensor Tensor::operator*(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] * tensor.arr[i];
    }
    return Tensor(new_arr);
}


Tensor Tensor::operator-(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] - tensor.arr[i];
    }
    return Tensor(new_arr);
}


Tensor Tensor::operator/(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }
    std::vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
        new_arr[i] = this->arr[i] / tensor.arr[i];
    }
    return Tensor(new_arr);
}
