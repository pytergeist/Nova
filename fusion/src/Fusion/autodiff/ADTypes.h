#ifndef AD_TYPES_H_
#define AD_TYPES_H_

#include <coroutine>
#include <cstddef>
#include <cstdint>

struct NodeID {
   std::int64_t idx;
   operator std::int64_t() const noexcept { return idx; }
};

struct ValueID {
   std::int64_t idx;

   operator std::int64_t() const noexcept { return idx; }

   bool operator==(const ValueID& other) const noexcept {
      return idx == other.idx;
   }
};

template <>
struct std::hash<ValueID> {
   std::size_t operator()(const ValueID& s) const noexcept {
      return std::hash<std::int64_t>{}(s.idx);
   }
};

struct GradSlotID {
   std::int64_t idx{-1};

   operator std::int64_t() const noexcept { return idx; }

   bool operator==(const GradSlotID& other) const noexcept {
      return idx == other.idx;
   }
};

template <>
struct std::hash<GradSlotID> {
   std::size_t operator()(const GradSlotID& s) const noexcept {
      return std::hash<std::int64_t>{}(s.idx);
   }
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
