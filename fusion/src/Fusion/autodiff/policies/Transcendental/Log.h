#ifndef LOG_H
#define LOG_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"
#include "../../AutodiffMode.h"

template <typename T>
struct Log {
    inline static constexpr std::string_view name = "Log";
    using In = AutodiffMeta<T>;
    using Out = AutodiffMeta<T>;
    using GradIn = AutodiffMeta<T>;
    using GradOut = AutodiffMeta<T>;

    Out forward(Context<T>& context, const In& input) {
        FUSION_CHECK(input.size() >= 1, "Log requires one inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        autodiff::NoGradGuard _;
        const Tensor<T>& a = input[0];
        context.save("c", a);
        Tensor<T> c = a.log();
        Out out;
        out.push_back(c);
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Log::backward expects exactly 1 upstream grad tensor");
        autodiff::NoGradGuard _;
        Tensor<T> g0 = std::move(grad_out[0]);
        FUSION_CHECK(!g0.empty(), "Log::backward: upstream grad is empty");
        const Tensor<T>& a = context.template load<Tensor<T>>("c");
        Tensor<T> ga = g0 / a;
        GradIn g;
        g.push_back(ga);
        return g;
    }
};

#endif // LOG_H
