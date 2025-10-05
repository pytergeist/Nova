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
        std::vector<T> c(input.at(0).size());
        context.save("a", input.at(0));
        context.save("b", input.at(1));
        for (size_t i = 0; i < input.at(0).size(); ++i) {
            c.at(i) = (input.at(1).at(i) * input.at(1).at(i));
        }
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        GradIn g;
        std::vector<T> c(grad_out.at(0).size());
        std::vector<T> d(grad_out.at(0).size());
        std::vector<T> a = context.template load<std::vector<T>>("a");
        std::vector<T> b = context.template load<std::vector<T>>("b");
        for (size_t i = 0; i < grad_out.at(0).size(); ++i) {
            const T& ai = a.at(i);
            const T& bi = b.at(i);
            const T& grad = grad_out.at(0).at(i);

            c.at(i) = grad * bi;
            d.at(i) = grad * ai;
        }
        g.push_back(std::move(c));
        g.push_back(std::move(d));
        return g;
    }
};

#endif // MULTIPLY_H
