#ifndef MATMUL_H
#define MATMUL_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"
#include "../../AutodiffMode.h"
#include "../../common/Log.h"

template <typename T>
struct MatMul {
    inline static constexpr std::string_view name = "MatMul";
    using In = AutodiffMeta<T>;
    using Out = AutodiffMeta<T>;
    using GradIn = AutodiffMeta<T>;
    using GradOut = AutodiffMeta<T>;

    Out forward(Context<T>& context, In& input) {
        autodiff::NoGradGuard _;
        FUSION_CHECK(input.size() >= 2, "MatMul requires two inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        FUSION_BOUNDS_CHECK(1, input.size());
        const auto& a = input[0];
        const auto& b = input[1];
        FUSION_LOGI("MatMul::forward  a.shape=", a.shape_str(),
            " b.shape=", b.shape_str(),
            " a.rank=", a.rank(), " b.rank=", b.rank(), " a.ndims=", a.ndims(), " b.ndims=", b.ndims());
        context.save("a", a);
        context.save("b", b);
        FUSION_CHECK(a.rank() >= 2 && b.rank() >= 2, "MatMul: rank must be >= 2");
        const auto K_a = a.shape_.back();
  		const auto K_b = b.shape_[b.rank_ - 2];
  		FUSION_CHECK(K_a == K_b, "MatMul: inner dims mismatch");
        Tensor<T> c = a.matmul(b);
        Out out;
        out.push_back(c);
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        autodiff::NoGradGuard _;
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "MatMul::backward expects exactly 1 upstream grad tensor");
        const Tensor<T>& a = context.template load<Tensor<T>>("a");
        const Tensor<T>& b = context.template load<Tensor<T>>("b");
        const Tensor<T>& g0 = grad_out[0];
        FUSION_CHECK(!g0.empty(), "MatMul::backward: upstream grad is empty");
    	auto bT = b.swapaxes(-1, -2);
    	auto aT = a.swapaxes(-1, -2);

    	Tensor<T> ga = g0.matmul(bT);
    	Tensor<T> gb = aT.matmul(g0);
        GradIn g;
        g.push_back(ga);
        g.push_back(gb);
        return g;
    }
};

#endif // MATMUL_H
