#include <vector>
#include "../Operation.h"


template <typename T>
struct Add {
  static constexpr std::string_view name = "Add";
  using In = BinaryType<T>;
  using Out = UnaryType<T>;
  using GradIn = BinaryType<T>;
  using GradOut = UnaryType<T>;

  Out forward(Context<T>& context, const In& input) {
    Out out;
    out.a.resize(input.a.size());
    for (size_t i = 0; i < input.a.size(); ++i) {
      out.a[i] = (input.a[i] + input.b[i]);
    }
    return out;
  };

  GradIn backward(Context<T>& context, GradOut& grad_out) {
    GradIn g;
    g.a = grad_out.a;
    g.b = grad_out.a;
    return g;
  }
};



template <typename T>
struct Subtract {
  static constexpr std::string_view name = "Subtract";
  using In = BinaryType<T>;
  using Out = UnaryType<T>;
  using GradIn = BinaryType<T>;
  using GradOut = UnaryType<T>;

  Out forward(Context<T>& context, const In& input) {
    Out out;
    out.a.resize(input.a.size());
    for (size_t i = 0; i < input.a.size(); ++i) {
      out.a[i] = (input.a[i] - input.b[i]);
    }
    return out;
  };

  GradIn backward(Context<T>& context, GradOut& grad_out) {
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

template <typename T>
struct Divide {
  static constexpr std::string_view name = "Divide";
  using In = BinaryType<T>;
  using Out = UnaryType<T>;
  using GradIn = BinaryType<T>;
  using GradOut = UnaryType<T>;

  Out forward(Context<T>& context, const In& input) {
    Out out;
    out.a.resize(input.a.size());
    context.save("vecs", input);
    for (size_t i = 0; i < input.a.size(); ++i) {
      out.a[i] = (input.a[i] / input.b[i]);
    }
    return out;
  };

  GradIn backward(Context<T>& context, GradOut& grad_out) {
    GradIn g;
    g.a.resize(grad_out.a.size());
    g.b.resize(grad_out.a.size());
    BinaryType<T> vecs = context.load("vecs");
    for (size_t i = 0; i < grad_out.a.size(); ++i) {
      const T& ai = vecs.a[i];
      const T& bi = vecs.b[i];
      const T& dyi = grad_out.a[i];

      g.a[i] = dyi / bi;
      g.b[i] = -dyi * ai / (bi * bi);
    }
    return g;
  }
};


template <typename T>
struct Multiply {
  static constexpr std::string_view name = "Multiply";
  using In = BinaryType<T>;
  using Out = UnaryType<T>;
  using GradIn = BinaryType<T>;
  using GradOut = UnaryType<T>;

  Out forward(Context<T>& context, const In& input) {
    Out out;
    out.a.resize(input.a.size());
    context.save("vecs", input);
    for (size_t i = 0; i < input.a.size(); ++i) {
      out.a[i] = (input.a[i] * input.b[i]);
    }
    return out;
  };

  GradIn backward(Context<T>& context, GradOut& grad_out) {
    GradIn g;
    g.a.resize(grad_out.a.size());
    g.b.resize(grad_out.a.size());
    BinaryType<T> vecs = context.load("vecs");
    for (size_t i = 0; i < grad_out.a.size(); ++i) {
      const T& ai = vecs.a[i];
      const T& bi = vecs.b[i];
      const T& grad = grad_out.a[i];

      g.a[i] = grad * bi;
      g.b[i] = grad * ai;
    }
    return g;
  }
};
