#ifndef REDUCTION_H
#define REDUCTION_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/AutodiffMeta.h"
#include "Fusion/autodiff/policies/Operation.h"

template <typename T> struct Sum {
   static constexpr std::string_view name = "Sum";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(!input.empty(), "Sum requires one inputs");
      const autodiff::NoGradGuard _;
      const ADTensor<T> &x = input.at(0);
      context.save("x", x);
      ADTensor<T> y = x.sum();
      Out out;
      out.push_back(y);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
         }
      FUSION_CHECK(grad_out.size() == 1,
                   "Sum::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      ADTensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Sum::backward: upstream grad is empty");
      const ADTensor<T> &x = context.template load<ADTensor<T>>("x");
      ADTensor<T> gx;
      if (g0.flat_size() == 1) {
         gx = ones_like(x) * g0;
      } else {
         gx = g0;
      }
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // REDUCTION_H
