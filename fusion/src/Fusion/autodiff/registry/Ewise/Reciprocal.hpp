#ifndef RECIPROCAL_HPP
#define RECIPROCAL_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T> struct Reciprocal {
   static constexpr std::string_view name = "Reciprocal";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) { // NOLINT
      const autodiff::NoGradGuard _;
      FUSION_CHECK(input.size() == 1, "Reciprocal requires one input");
      const RawTensor<T> &x = input.at(0);
      RawTensor<T> z = x.reciprocal();
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) { // NOLINT
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Add::backward expects exactly 1 upstream grad tensor");
      RawTensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Reciprocal::backward: upstream grad is empty");
      const autodiff::NoGradGuard _;
      RawTensor<T> gx = g0.pow(2) * -1;
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // RECIPROCAL_HPP
