#ifndef FUSION_CORE_OP_SCHEMA_OP_SCHEMA_H_
#define FUSION_CORE_OP_SCHEMA_OP_SCHEMA_H_

#include <cstddef>

enum class OpCategory {
   EwiseUnary,
   EwiseBinary,
   Reduction,
   Contraction,
   Movement,
   Topology
};

enum class ArityKind { Fixed, Variadic };

struct AritySpec {
   ArityKind kind;
   std::size_t arity;
};

enum class MutationKind {
   OutOfPlace,
   InplaceAllowed,
   InplaceRequired,
   Accumulate,
   Scatter
};

struct OpSchema {
   OpCategory category;
   AritySpec inputs;
   AritySpec outputs;
   MutationKind mutation;
};

#endif // FUSION_CORE_OP_SCHEMA_OP_SCHEMA_H_