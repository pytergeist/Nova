#ifndef DIVIDE_H
#define DIVIDE_H

#include <vector>
#include <string_view>
#include "../Operation.h"

template <typename T>
struct Divide {
    inline static constexpr std::string_view name = "Divide";
    using In = BinaryType<T>;
    using Out = UnaryType<T>;
    using GradIn = BinaryType<T>;
    using GradOut = UnaryType<T>;

    Out forward(Context& context, const In& input) {
        Out out;
        out.a.resize(input.a.size());
        context.save("a", input.a);
        context.save("b", input.b);
        for (size_t i = 0; i < input.a.size(); ++i) {
            out.a[i] = (input.a[i] / input.b[i]);
        }
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        GradIn g;
        g.a.resize(grad_out.a.size());
        g.b.resize(grad_out.a.size());
        std::vector<T> a = context.template load<std::vector<T>>("a");
        std::vector<T> b = context.template load<std::vector<T>>("b");
        for (size_t i = 0; i < grad_out.a.size(); ++i) {
            const T& ai = a[i];
            const T& bi = b[i];
            const T& dyi = grad_out.a[i];

            g.a[i] = dyi / bi;
            g.b[i] = -dyi * ai / (bi * bi);
        }
        return g;
    }
};

#endif // DIVIDE_H
