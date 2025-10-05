#ifndef SUBTRACT_H
#define SUBTRACT_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"

template <typename T>
struct Subtract {
  inline static constexpr std::string_view name = "Subtract";
  using In = MultiTensor<T>;
  using Out = MultiTensor<T>;
  using GradIn = MultiTensor<T>;
  using GradOut = MultiTensor<T>;

  Out forward(Context& context, const In& input) {
    std::vector<T> c(input[0].size());
    for (size_t i = 0; i < input[0].size(); ++i) {
      c[i] = (input[0][i] - input[1][i]);
    }
    Out out;
    out.push_back(c);
    return out;
  };

  GradIn backward(Context& context, GradOut& grad_out) {
    const auto& c = grad_out[0];
    std::vector<T> d(c.size());
    for (size_t i = 0; i < grad_out[0].size(); ++i) {
      const T& bi = -grad_out[0][i];
      d[i] = bi;
    }
    GradIn g;
    g.push_back(c);
    g.push_back(d);
    return g;
  }
};

#endif // SUBTRACT_H
