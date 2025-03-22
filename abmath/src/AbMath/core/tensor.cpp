#include <iostream>
#include <vector>

using namespace std;

class Tensor {
public:
    vector<double> arr;

    explicit Tensor(const vector<double> &data) : arr(data) { }

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

    Tensor &operator+(const Tensor &tensor) const {
        vector<double> new_arr(tensor.arr.size());
        vector<double> const other_arr = tensor.arr;
        for (size_t i = 0; i < this->arr.size(); ++i) {
            new_arr[i] = this->arr[i] + other_arr[i];
        }
        return *new Tensor(new_arr);
    }

    Tensor &operator*(const Tensor &tensor) const {
        vector<double> new_arr(tensor.arr.size());
        vector<double> const other_arr = tensor.arr;
        for (size_t i = 0; i < this->arr.size(); ++i) {
            new_arr[i] = this->arr[i] * other_arr[i];
        }
        return *new Tensor(new_arr);
    }

    Tensor &operator-(const Tensor &tensor) const {
        vector<double> new_arr(tensor.arr.size());
        vector<double> const other_arr = tensor.arr;
        for (size_t i = 0; i < this->arr.size(); ++i) {
            new_arr[i] = this->arr[i] - other_arr[i];
        }
        return *new Tensor(new_arr);
    }

    Tensor &operator/(const Tensor &tensor) const {
        vector<double> new_arr(tensor.arr.size());
        vector<double> const other_arr = tensor.arr;
        for (size_t i = 0; i < this->arr.size(); ++i) {
            new_arr[i] = this->arr[i] / other_arr[i];
        }
        return *new Tensor(new_arr);
    }

};

int main() {
    Tensor const tensor1({1.0, 2.0, 3.0, 4.0, 5.0});
    Tensor const tensor2({1.0, 2.0, 3.0, 4.0, 5.0});
    Tensor const& tensor3 = tensor1 - tensor2;
    cout << tensor3 << endl;
    return 0;
}
