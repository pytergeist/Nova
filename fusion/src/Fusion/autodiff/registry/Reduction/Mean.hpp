#ifndef MEAN_HPP
#define MEAN_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T> struct Mean {
   static constexpr std::string_view name = "Mean";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(!input.empty(), "Mean requires one inputs");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      context.save("x", x);
      const ReductionParam &p =
          std::any_cast<const ReductionParam &>(input.op_param);
      RawTensor<T> y = x.mean(p.reduction_axis, p.keepdim);
      Out out;
      out.push_back(x);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Mean::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      RawTensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Mean::backward: upstream grad is empty");
      const RawTensor<T> &x = context.template load<RawTensor<T>>("x");
      RawTensor<T> gx;
      if (g0.flat_size() == 1) {
         gx = ones_like(x) * g0;
      } else {

         gx = g0;
      }
      RawTensor<T> gy = gx * (1.0 / x.size());
      GradIn g;
      g.push_back(gy);
      return g;
   }
};

#endif // MEAN_HPP
