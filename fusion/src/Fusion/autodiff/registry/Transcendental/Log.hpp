// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef LOG_HPP
#define LOG_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T> struct Log {
   static constexpr std::string_view name = "Log";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(!input.empty(), "Log requires one inputs");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      context.save("x", x);
      RawTensor<T> y = x.log();
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
      RawTensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Log::backward: upstream grad is empty");
      const RawTensor<T> &x = context.template load<RawTensor<T>>("x");
      RawTensor<T> gx = g0 / x;
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // LOG_HPP
