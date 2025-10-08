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
    FUSION_CHECK(input.size() >= 2, "Subtract requires two inputs");
    FUSION_BOUNDS_CHECK(0, input.size());
    FUSION_BOUNDS_CHECK(1, input.size());
    const auto& a = input[0];
    const auto& b = input[1];
    FUSION_CHECK(a.size() == b.size(), "Subtract: input size mismatch");
    std::vector<T> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      c[i] = a[i] - b[i];
    }
    Out out;
    out.push_back(std::move(c));
    return out;
  };

  GradIn backward(Context& context, GradOut& grad_out) {
    if (grad_out.size() == 0) return {};
    FUSION_CHECK(grad_out.size() == 1, "Subtract::backward expects exactly 1 upstream grad tensor");
    const auto& g0 = grad_out[0];
    std::vector<T> g1(g0.size());
    FUSION_CHECK(!g0.empty(), "Subtract::backward: upstream grad is empty");
    for (size_t i = 0; i < g0.size(); ++i) {
      const T& bi = -g0[i];
      g1[i] = bi;
    }
    GradIn g;
    g.push_back(std::move(g0));
    g.push_back(std::move(g1));
    return g;
  }
};

#endif // SUBTRACT_H
