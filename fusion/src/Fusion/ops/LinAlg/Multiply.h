#ifndef MULTIPLY_H
#define MULTIPLY_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"


template <typename T>
struct Multiply {
    inline static constexpr std::string_view name = "Multiply";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context& context, const In& input) {
        std::vector<T> c(input[0].size());
        context.save("a", input[0]);
        context.save("b", input[1]);
        for (size_t i = 0; i < input[0].size(); ++i) {
            c[i] = (input[1][i] * input[1][i]);
        }
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        GradIn g;
        std::vector<T> c(grad_out[0].size());
        std::vector<T> d(grad_out[0].size());
        std::vector<T> a = context.template load<std::vector<T>>("a");
        std::vector<T> b = context.template load<std::vector<T>>("b");
        for (size_t i = 0; i < grad_out[0].size(); ++i) {
            const T& ai = a[i];
            const T& bi = b[i];
            const T& grad = grad_out[0][i];

            c[i] = grad * bi;
            d[i] = grad * ai;
        }
        g.push_back(std::move(c));
        g.push_back(std::move(d));
        return g;
    }
};

#endif // MULTIPLY_H
