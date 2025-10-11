#ifndef ADD_H
#define ADD_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"
#include "../../common/Checks.h"


template <typename T>
struct Add {
    inline static constexpr std::string_view name = "Add";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context<T>& context, const In& input) {
        FUSION_CHECK(input.size() >= 2, "Add requires two inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        FUSION_BOUNDS_CHECK(1, input.size());
    	const auto& a = input[0];
    	const auto& b = input[1];
    	FUSION_CHECK(a.size() == b.size(), "Add: input size mismatch");
        Tensor<T> c = a + b;
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {}; // TODO: Make a macro??? or helper func
        FUSION_CHECK(grad_out.size() == 1, "Add::backward expects exactly 1 upstream grad tensor");
        Tensor<T>& g0 = grad_out[0];
        FUSION_CHECK(!g0.empty(), "Add::backward: upstream grad is empty");
        Tensor<T> ga = g0;
        Tensor<T> gb = g0;
        GradIn g;
        g.data.reserve(2);
        g.push_back(std::move(ga));
        g.push_back(std::move(gb));
        return g;
    }
};

#endif // ADD_H
