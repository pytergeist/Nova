#ifndef FUSION_ALLOC_ALLOC_TYPES_H
#define FUSION_ALLOC_ALLOC_TYPES_H

#include <cstddef>

struct Alignment {
   std::size_t value;
   operator std::size_t() const noexcept { return value; }
};

#endif // ALLOC_TYPES_H
