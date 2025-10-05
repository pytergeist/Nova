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
    std::vector<T> c(input.at(0).size());
    for (size_t i = 0; i < input.at(0).size(); ++i) {
      c.at(i) = (input.at(0).at(i) - input.at(1).at(i));
    }
    Out out;
    out.push_back(c);
    return out;
  };

  GradIn backward(Context& context, GradOut& grad_out) {
    const auto& c = grad_out.at(0);
    std::vector<T> d(c.size());
    for (size_t i = 0; i < grad_out.at(0).size(); ++i) {
      const T bi = -grad_out.at(0).at(i);
      d.at(i) = bi;
    }
    GradIn g;
    g.push_back(std::move(c));
    g.push_back(std::move(d));
    return g;
  }
};

#endif // SUBTRACT_H
