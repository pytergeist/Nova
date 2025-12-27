#ifndef EXP_HPP
#define EXP_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T> struct Exp {
   static constexpr std::string_view name = "Exp";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(!input.empty(), "Exp requires one inputs");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      context.save("x", x);
      RawTensor<T> y = x.exp();
      Out out;
      out.push_back(y);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Exp::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      RawTensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Exp::backward: upstream grad is empty");
      const RawTensor<T> &x = context.template load<RawTensor<T>>("x");
      RawTensor<T> gx = g0 * x.exp();
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // EXP_HPP
