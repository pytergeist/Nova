#ifndef EXP_H
#define EXP_H

#include <vector>
#include <string_view>
#include "../Operation.h"


template <typename T>
struct Exp {
    inline static constexpr std::string_view name = "Exp";
    using In = UnaryType<T>;
    using Out = UnaryType<T>;
    using GradIn = UnaryType<T>;
    using GradOut = UnaryType<T>;

    Out forward(Context& context, const In& input) {
        Out out;
        out.a.resize(input.a.size());
        for (size_t i = 0; i < input.a.size(); ++i) {
            out.a[i] = std::exp(input.a[i]);
        }
        context.save("c", out.a);
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        GradIn g;
        g.a.resize(grad_out.a.size());
        std::vector<T> a = context.template load<std::vector<T>>("c");
        for (size_t i = 0; i < grad_out.a.size(); ++i) {
            const T& ai = a[i];
            const T& dyi = grad_out.a[i];
            g.a[i] = dyi * ai;
        }
        return g;
    }
};

#endif // EXP_H
