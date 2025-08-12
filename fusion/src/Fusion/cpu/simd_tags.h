#ifndef SIMD_TAGS_H
#define SIMD_TAGS_H

#pragma once

struct AddSIMD {
    template<typename U>
    constexpr U operator()(U a, U b) const noexcept {
        return a + b;
    }
};


struct SubtractSIMD {
    template<typename U>
    constexpr U operator()(U a, U b) const noexcept {
        return a - b;
    }
};



struct DivideSIMD {
    template<typename U>
    constexpr U operator()(U a, U b) const noexcept {
        return a / b;
    }
};


struct MultiplySIMD {
    template<typename U>
    constexpr U operator()(U a, U b) const noexcept {
        return a * b;
    }
};


struct MaximumSIMD {
    template<typename U>
    constexpr U operator()(U a, U b) const noexcept {
        return a > b;
    }
};


#endif // SIMD_TAGS_H
