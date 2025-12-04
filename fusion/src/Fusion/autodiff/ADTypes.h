#ifndef _AD_TYPES_H
#define _AD_TYPES_H

#include <cstdint>

struct ValueID {
    int32_t idx;
};
struct NodeID {
    int32_t idx;
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
    size_t out_slot;
};

struct ConsumerInfo {
    NodeID nid;
    size_t in_slot;
};

#endif // _AD_TYPES_H
