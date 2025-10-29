#ifndef POW_H
#define POW_H

#include "../../AutodiffMode.h"
#include "../../Traits.h"
#include "../Operation.h"
#include <string_view>
#include <vector>

template <typename T> struct Pow {
   inline static constexpr std::string_view name = "Pow";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 2, "Pow requires two inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      FUSION_BOUNDS_CHECK(1, input.size());
      autodiff::NoGradGuard _;
      const Tensor<T> &a = input.at(0);
      const Tensor<T> &b = input.at(1);
      FUSION_ALLOW_SCALAR_BINARY(a, b);
      context.save("a", input[0]);
      context.save("b", input[1]);
      Tensor<T> c = a.pow(b);
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.size() == 0)
         return {};
      FUSION_CHECK(grad_out.size() == 1,
                   "Pow::backward expects exactly 1 upstream grad tensor");
      autodiff::NoGradGuard _;
      const Tensor<T> &a = context.template load<Tensor<T>>("a");
      const Tensor<T> &b = context.template load<Tensor<T>>("b");
      const auto &g0 = grad_out[0];
      FUSION_CHECK(!g0.empty(), "Pow::backward: upstream grad is empty");
      Tensor<T> ga = (b * a.pow(b - 1)) * g0;
      Tensor<T> gb = (a.pow(b) * a.log()) * g0;
      GradIn g;
      g.push_back(ga);
      g.push_back(gb);
      return g;
   }
};

#endif // POW_H
