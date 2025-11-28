#ifndef ADD_H
#define ADD_H

#include <string_view>
#include <vector>
#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/Traits.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Add {
   inline static constexpr std::string_view name = "Add";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      autodiff::NoGradGuard _;
      FUSION_CHECK(input.size() >= 2, "Add requires two inputs");
      FUSION_BOUNDS_CHECK(0, input.size());
      FUSION_BOUNDS_CHECK(1, input.size());
      const auto &a = input[0];
      const auto &b = input[1];
      //        FUSION_ALLOW_SCALAR_BINARY(a, b);
      Tensor<T> c = a + b;
      Out out;
      out.push_back(c);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      autodiff::NoGradGuard _;
      if (grad_out.size() == 0)
         return {}; // TODO: Make a macro??? or helper func
      FUSION_CHECK(grad_out.size() == 1,
                   "Add::backward expects exactly 1 upstream grad tensor");
      Tensor<T> &g0 = grad_out[0];
      FUSION_CHECK(!g0.empty(), "Add::backward: upstream grad is empty");
      Tensor<T> ga = g0;
      Tensor<T> gb = g0;
      GradIn g;
      g.data.reserve(2);
      g.push_back(ga);
      g.push_back(gb);
      return g;
   }
};

#endif // ADD_H
