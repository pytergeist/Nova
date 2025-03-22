#include <iostream>
#include <vector>

using namespace std;

class Tensor {
public:
  vector<double> arr;

  explicit Tensor(const vector<double> &data) : arr(data) {}

  friend ostream &operator<<(ostream &os, const Tensor &tensor) {
    os << "Tensor(";
    for (size_t i = 0; i < tensor.arr.size(); ++i) {
      os << tensor.arr[i];
      if (i < tensor.arr.size() - 1)
        os << ", ";
    }
    os << ")";
    return os;
  }

  Tensor operator+(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
      throw std::invalid_argument("Tensor sizes do not match");
    }
    vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
      new_arr[i] = this->arr[i] + tensor.arr[i];
    }
    return *new Tensor(new_arr);
  }

  Tensor operator*(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
      throw std::invalid_argument("Tensor sizes do not match");
    }
    vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
      new_arr[i] = this->arr[i] * tensor.arr[i];
    }
    return Tensor(new_arr);
  }

  Tensor operator-(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
      throw std::invalid_argument("Tensor sizes do not match");
    }
    vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
      new_arr[i] = this->arr[i] - tensor.arr[i];
    }
    return Tensor(new_arr);
  }

  Tensor operator/(const Tensor &tensor) const {
    if (arr.size() != tensor.arr.size()) {
      throw std::invalid_argument("Tensor sizes do not match");
    }
    vector<double> new_arr(tensor.arr.size());
    for (size_t i = 0; i < this->arr.size(); ++i) {
      new_arr[i] = this->arr[i] / tensor.arr[i];
    }
    return Tensor(new_arr);
  }
};

int main() {
  Tensor const tensor1({1.0, 2.0, 3.0, 4.0, 5.0});
  Tensor const tensor2({1.0, 2.0, 3.0, 4.0, 5.0});
  Tensor const &tensor3 = tensor1 - tensor2;
  cout << tensor3 << endl;
  return 0;
}
