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

  Out forward(Context<T>& context, const In& input) {
    FUSION_CHECK(input.size() >= 2, "Subtract requires two inputs");
    FUSION_BOUNDS_CHECK(0, input.size());
    FUSION_BOUNDS_CHECK(1, input.size());
    const auto& a = input[0];
    const auto& b = input[1];
    FUSION_CHECK(a.size() == b.size(), "Subtract: input size mismatch");
    Tensor<T> c = a - b;
    Out out;
    out.push_back(c);
    return out;
  };

  GradIn backward(Context<T>& context, GradOut& grad_out) {
    if (grad_out.size() == 0) return {};
    FUSION_CHECK(grad_out.size() == 1, "Subtract::backward expects exactly 1 upstream grad tensor");
    Tensor<T>& g0 = grad_out[0];
    FUSION_CHECK(!g0.empty(), "Subtract::backward: upstream grad is empty");
    Tensor<T> g1 = zeros_like(g0) - g0;
    GradIn g;
    g.push_back(g0);
    g.push_back(g1);
    return g;
  }
};

#endif // SUBTRACT_H
