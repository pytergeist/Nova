#ifndef SUBTRACT_H
#define SUBTRACT_H

#include <vector>
#include "../Operation.h"

template <typename T>
struct Subtract {
  static constexpr std::string_view name = "Subtract";
  using In = BinaryType<T>;
  using Out = UnaryType<T>;
  using GradIn = BinaryType<T>;
  using GradOut = UnaryType<T>;

  Out forward(Context& context, const In& input) {
    Out out;
    out.a.resize(input.a.size());
    for (size_t i = 0; i < input.a.size(); ++i) {
      out.a[i] = (input.a[i] - input.b[i]);
    }
    return out;
  };

  GradIn backward(Context& context, GradOut& grad_out) {
    GradIn g;
    g.a = grad_out.a;
    g.b.resize(g.a.size());
    for (size_t i = 0; i < grad_out.a.size(); ++i) {
      const T& bi = -g.a[i];
      g.b[i] = bi;
    }
    return g;
  }
};

#endif // SUBTRACT_H
