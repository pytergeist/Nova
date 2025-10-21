// SwapAxes.h
#ifndef SWAPAXES_POLICY_H
#define SWAPAXES_POLICY_H

#include <string_view>
#include "../../Traits.h"
#include "../../Operation.h"
#include "../../AutodiffMode.h"
#include "../../common/Checks.h"
#include "../../../Tensor.h"
#include "../../kernels/Serial.h"

template <typename T>
struct SwapAxes {
    inline static constexpr std::string_view name = "SwapAxes";
    using In      = AutodiffMeta<T>;
    using Out     = AutodiffMeta<T>;
    using GradIn  = AutodiffMeta<T>;
    using GradOut = AutodiffMeta<T>;

    Out forward(Context<T>& context, In& input) {
        autodiff::NoGradGuard _;
        FUSION_CHECK(input.size() == 1, "SwapAxes requires exactly one input");
        const auto& x = input[0];

        int a1 = input.template get_param<int>("axis1");
        int a2 = input.template get_param<int>("axis2");

        a1 = serial_ops::normalise_axis(a1, x.rank());
        a2 = serial_ops::normalise_axis(a2, x.rank());
        FUSION_CHECK(a1 != a2, "SwapAxes: axes must be different");

        context.save("axis1", a1);
        context.save("axis2", a2);

        Tensor<T> y = x.swapaxes(a1, a2);  // raw kernel is fine here
        Out out;
        out.push_back(y);
        return out;
    }

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        autodiff::NoGradGuard _;
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "SwapAxes::backward expects 1 upstream grad");

        const int a1 = context.template load<int>("axis1");
        const int a2 = context.template load<int>("axis2");

        const Tensor<T>& gy = grad_out[0];
        FUSION_CHECK(!gy.empty(), "SwapAxes::backward: upstream grad is empty");

        GradIn g;
        g.push_back(gy.swapaxes(a1, a2));
        return g;
    }
};

#endif // SWAPAXES_POLICY_H
