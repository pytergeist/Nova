#ifndef AD_TYPES_H_
#define AD_TYPES_H_

#include <cstdint>
#include <cstddef>

struct ValueID {
   std::int64_t idx;
   operator std::int64_t() const noexcept { return idx; }
};
struct NodeID {
   std::int64_t idx;
   operator std::int64_t() const noexcept { return idx; }
};

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
// bugprone-easily-swappable-parameters)
struct Edge {
   NodeID src;
   NodeID dst;
   Edge(NodeID src = NodeID{-1}, NodeID dst = NodeID{-1})
       : src(src), dst(dst) {};
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
// bugprone-easily-swappable-parameters)

struct ProducerInfo {
   NodeID nid;
   std::size_t out_slot;
};

struct ConsumerInfo {
   NodeID nid;
   std::size_t in_slot;
};

#endif // AD_TYPES_H_
