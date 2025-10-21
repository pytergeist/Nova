#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"


template <typename T>
struct Transpose {
    inline static constexpr std::string_view name = "Transpose";
    using In = AutodiffMeta<T>;
    using Out = AutodiffMeta<T>;
    using GradIn = AutodiffMeta<T>;
    using GradOut = AutodiffMeta<T>;

    Out forward(Context<T>& context, const In& input) {
        FUSION_CHECK(input.size() >= 1, "Transpose requires one inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        const auto& a = input[0];
        Tensor<T> c = a.transpose();
        Out out;
        out.push_back(c);
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Transpose::backward expects exactly 1 upstream grad tensor");
        Tensor<T> g0 = std::move(grad_out[0]);
        FUSION_CHECK(!g0.empty(), "Transpose::backward: upstream grad is empty");
        Tensor<T> g1 = g0.transpose();
        GradIn g;
        g.push_back(g1);
        return g;
    }
};

#endif // TRANSPOSE_H
