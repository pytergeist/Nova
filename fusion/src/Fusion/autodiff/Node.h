#ifndef _NODE_H
#define _NODE_H

#include <memory>

template <typename T, class Op, class Ctx>
class Node {
    public:
      std::shared_ptr<Op> op;
      std::shared_ptr<Ctx> ctx;
      Node(std::shared_ptr<Op> op, std::shared_ptr<Ctx> ctx) : op(std::move(op)), ctx(std::move(ctx)) {}

};

#endif // _NODE_H
