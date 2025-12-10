#ifndef LOG_H
#define LOG_H

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.h"
#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/autodiff/policies/Operation.h"
#include "Fusion/common/Checks.h"

template <typename T> struct Log {
   static constexpr std::string_view name = "Log";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(!input.empty(), "Log requires one inputs");
      const autodiff::NoGradGuard _;
      const ADTensor<T> &x = input.at(0);
      context.save("x", x);
      ADTensor<T> y = x.log();
      Out out;
      out.push_back(y);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Log::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      ADTensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Log::backward: upstream grad is empty");
      const ADTensor<T> &x = context.template load<ADTensor<T>>("x");
      ADTensor<T> gx = g0 / x;
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // LOG_H
