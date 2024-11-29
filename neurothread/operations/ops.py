from .operation import Operation
from neurothread.autodiff.autodiff import AutoDiff

# Defining operations

add_op = Operation(
    "add",
    lambda a, b: a.data + b.data,
    lambda a, b, grad: (
        AutoDiff.generic_backward_func(a, grad),
        AutoDiff.generic_backward_func(b, grad),
    ),
)

subtract_op = Operation(
    "subtract",
    lambda a, b: a.data - b.data,
    lambda a, b, grad: (
        AutoDiff.generic_backward_func(a, grad),
        AutoDiff.generic_backward_func(b, -grad),
    ),
)

right_subtract_op = Operation(
    "right_subtract",
    lambda a, b: a.data - b.data,
    lambda a, b, grad: (
        AutoDiff.generic_backward_func(b, grad),
        AutoDiff.generic_backward_func(a, -grad),
    ),
)


