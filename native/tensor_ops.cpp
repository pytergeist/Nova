#include <iostream>
#include <vector>
#include <variant>
#include <stdexcept>
using namespace std;

using VectorOrScalar = variant<vector<double>, double>;


VectorOrScalar add(const VectorOrScalar& a, const VectorOrScalar& b) {
  // Both a and b are vectors
  if (holds_alternative<vector<double>>(a) && holds_alternative<vector<double>>(b)) {
    const auto va = get<vector<double>>(a);
    const auto vb = get<vector<double>>(b);
    if (va.size() != vb.size()) {
      throw invalid_argument(string("Vectors must be of the same shape, but got ") +
                         to_string(va.size()) + " and " + to_string(vb.size()));
    }
    vector<double> result(va.size());
    for (size_t i = 0; i < va.size(); ++i) {
      result[i] = va[i] + vb[i];
    }
    return result;
  }
  // a is a vector, b is a scalar
  if (holds_alternative<vector<double>>(a) && holds_alternative<double>(b)) {
    const auto va = get<vector<double>>(a);
    const auto b_scalar = get<double>(b);
    vector<double> result(va.size());
    for (size_t i = 0; i < va.size(); i++) {
      result[i] = va[i] + b_scalar;
    }
    return result;
  }

  // a is a scalar, b is a vector
  if (holds_alternative<double>(a) && holds_alternative<vector<double>>(b)) {
    const auto a_scalar = get<double>(a);
    const auto vb = get<vector<double>>(b);
    vector<double> result(vb.size());
    for (size_t i = 0; i < vb.size(); i++) {
      result[i] = a_scalar + vb[i];
    }
    return result;
  }

  // a and b are scalars
  if (holds_alternative<double>(a) && holds_alternative<double>(b)) {
    const auto a_scalar = get<double>(a);
    const auto b_scalar = get<double>(b);
    return a_scalar + b_scalar;
  }
  throw invalid_argument("Unsupported operand types for addition.");
}



int main() {
  vector<double> v1 ={1, 2, 3};
  double v2 = 2;
  VectorOrScalar result = add(v1, v2);
  const auto& v3 = get<vector<double>>(result);
  for (auto i : v3) {
    cout << i << " ";
  }
  cout << endl;
  return 0;
}
