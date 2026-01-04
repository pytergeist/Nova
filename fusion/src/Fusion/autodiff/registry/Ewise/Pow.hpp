// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef POW_HPP
#define POW_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T> struct Pow {
   static constexpr std::string_view name = "Pow";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 2, "Pow requires two inputs");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      const RawTensor<T> &y = input.at(1);
      FUSION_ALLOW_SCALAR_BINARY(x, y);
      context.save("x", x);
      context.save("y", y);
      RawTensor<T> z = x.pow(y);
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Pow::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = context.template load<RawTensor<T>>("x");
      const RawTensor<T> &y = context.template load<RawTensor<T>>("y");
      const RawTensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Pow::backward: upstream grad is empty");
      RawTensor<T> gx = (y * x.pow(y - 1)) * g0;
      RawTensor<T> gy = (x.pow(y) * x.log()) * g0;
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // POW_HPP
