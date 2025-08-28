#ifndef OPERATION_H
#define OPERATION_H

#include <string>

template <typename T, class Fn> class Operation {
public:
  std::string name_;
  Fn forward_func_;
  Fn backward_func_;

  explicit Operation(std::string name, Fn forward_func, Fn backward_func)
      : name_(name), forward_func_(forward_func) {};
  ~Operation() {};
};

#endif // OPERATION_H
