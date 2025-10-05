#ifndef DIVIDE_H
#define DIVIDE_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"

template <typename T>
struct Divide {
    inline static constexpr std::string_view name = "Divide";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context& context, const In& input) {
        context.save("a", input.at(0));
        context.save("b", input.at(1));
        std::vector<T> c(input.at(0).size());
        for (size_t i = 0; i < input.at(0).size(); ++i) {
            c.at(i) = (input.at(0).at(i) / input.at(0).at(i));
        }
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        std::vector<T> a = context.template load<std::vector<T>>("a");
        std::vector<T> b = context.template load<std::vector<T>>("b");
        std::vector<T> c(grad_out.at(0).size());
        std::vector<T> d(grad_out.at(0).size());
        for (size_t i = 0; i < grad_out.at(0).size(); ++i) {
            const T& ai = a.at(i);
            const T& bi = b.at(i);
            const T& dyi = grad_out.at(0).at(i);

            c.at(i) = dyi / bi;
            d.at(i) = -dyi * ai / (bi * bi);
        }
        GradIn g;
        g.push_back(std::move(c));
        g.push_back(std::move(d));
        return g;
    }
};

#endif // DIVIDE_H
