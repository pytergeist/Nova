#ifndef FUSION_PHYSICS_CORE_PHYSICS_DESCS_H
#define FUSION_PHYSICS_CORE_PHYSICS_DESCS_H

#include <cstddef>
#include <cstdint>

#include "Fusion/core/fuir/Descs.h"

struct ParticlesAoSoADesc {
   std::int64_t N{0};
   std::size_t E{0};
   std::size_t tile{0};
   std::size_t dim{0};

   OperandDescription x_desc;
   OperandDescription f_desc;
   OperandDescription v_desc;

   std::size_t itemsize{0};
};

#endif // FUSION_PHYSICS_CORE_PHYSICS_DESCS_H