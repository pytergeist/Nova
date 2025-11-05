#ifndef DTYPE_H
#define DTYPE_H

#include <cstddef>


enum class DType { Float32, Float64, Int32, Int64 };

inline std::size_t get_dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
        case DType::Float64:
            return sizeof(double);
        case DType::Int32:
            return sizeof(int32_t);
        case DType::Int64:
            return sizeof(int64_t);
    }
}

#endif // DTYPE_H
