# registry.py
import numpy as np

from .operation import Operation


########################
# ADD
########################
def add_backward(result, a, b, grad_output):
    """C = a + b derivative wrt a = grad_output derivative wrt b = grad_output."""
    return (grad_output, grad_output)


add_op = Operation("__add__", lambda a, b: a + b, add_backward)


########################
# SUBTRACT
########################
def subtract_backward(result, a, b, grad_output):
    """
    c = a - b
    dc/da = +1 * grad_output
    dc/db = -1 * grad_output
    """
    return grad_output, -grad_output


subtract_op = Operation("__sub__", lambda a, b: a - b, subtract_backward)


########################
# RIGHT-SUBTRACT
########################
def right_subtract_backward(result, a, b, grad_output):
    """
    c = b - a
    => derivative wrt 'a' = -grad_output
       derivative wrt 'b' = +grad_output
    (assuming parents are (a, b) in that order)
    """
    return (-grad_output, grad_output)


right_subtract_op = Operation(
    "__rsub__",
    lambda a, b: b - a,
    right_subtract_backward,
)


########################
# DIVIDE
########################
def divide_backward(result, a, b, grad_output):
    grad_a = grad_output / b.value
    grad_b = -grad_output * a.value / (b.value**2)
    return (grad_a, grad_b)


divide_op = Operation("__truediv__", lambda a, b: a / b, divide_backward)


def generic_transpose(x):
    return x if x.ndim < 2 else x.swapaxes(-1, -2)


def matmul_backward(
    result, a, b, grad_output
):  # TODO: This function needs to be made generic
    A = a.value
    B = b.value
    dZ = grad_output
    if dZ.ndim == 1:
        dZ = dZ[:, None]
    if B.ndim == 1:
        B = B[:, None]

    grad_a = dZ @ generic_transpose(B)
    grad_b = generic_transpose(A) @ dZ

    if a.value.ndim == 1:
        grad_a = grad_a.ravel()
    if b.value.ndim == 1:
        grad_b = grad_b.ravel()

    return grad_a, grad_b


matmul_op = Operation(
    "__matmul__",
    lambda a, b: a @ b,  # forward pass
    matmul_backward,  # backward pass
)


def sum_backward(result, a, grad_output):
    """
    Backward pass:
      result = sum(a)

    If 'result' is a scalar, 'grad_output' is also scalar (or shape () in NumPy).
    The gradient wrt 'a' is just grad_output * ones_like(a.value).
    """
    g = grad_output if grad_output is not None else np.array([1.0])
    return np.ones_like(a.value) * float(np.array(g).ravel()[0])


sum_op = Operation(
    name="sum",
    forward_func=lambda a: a.sum(),
    backward_func=sum_backward,
)


def maximum_backward(result, a, b, grad_output):
    """
    result = np.maximum(a.value, b.value)
    We take a subgradient approach:
      grad wrt a = grad_output if a.value >= b.value, else 0
      grad wrt b = grad_output if b.value >  a.value, else 0
    (Ties are broken to favor 'a' in this version.)
    """
    grad_a = grad_output * (a.value >= b.value)
    grad_b = grad_output * (b.value > a.value)
    return (grad_a, grad_b)


maximum_op = Operation(
    name="maximum",
    forward_func=lambda a, b: np.maximum(a.data, b.data),
    backward_func=maximum_backward,
)


def transpose_backward(result, a, grad_output):
    """
    For a transpose operation, the backward pass simply transposes the gradient.
    Since the forward pass is: result = transpose(a.data),
    the derivative wrt 'a' is simply: transpose(grad_output).
    """
    return (grad_output.T,)


transpose_op = Operation(
    name="transpose",
    forward_func=lambda a: a.T,
    backward_func=transpose_backward,
)


def multiply_backward(result, a, b, grad_output):
    """
    c = a * b
    dc/da = b * grad_output
    dc/db = a * grad_output
    """
    return b.value * grad_output, a.value * grad_output


multiply_op = Operation(
    name="__mul__",
    forward_func=lambda a, b: a * b,
    backward_func=multiply_backward,
)


right_multiply_op = Operation(
    name="__mul__",
    forward_func=lambda a, b: b * a,
    backward_func=multiply_backward,
)


def power_backward(result, a, b, grad_output):
    """
    c = a ** b
    dc/da = b * a ** (b - 1) * grad_output
    dc/db = a ** b * log(a) * grad_output
    """
    grad_a = (b.value * a.value ** (b.value - 1)) * grad_output
    grad_b = (a.value**b.value * np.log(a.value)) * grad_output
    return grad_a, grad_b


power_op = Operation(
    name="__pow__",
    forward_func=lambda a, b: a**b,
    backward_func=power_backward,
)


def exponential_backward(result, a, grad_output):
    """
    c = exp(a)
    dc/da = exp(a) * grad_output
    """
    return (result.value * grad_output,)


exponential_op = Operation(
    name="exp",
    forward_func=lambda a: a.exp(),
    backward_func=exponential_backward,
)


def log_backward(result, a, grad_output):
    """
    c = ln(a)
    dc/da = 1/a * grad_output
    """
    return (grad_output / a.value,)


log_op = Operation(
    name="log",
    forward_func=lambda a: a.log(),
    backward_func=log_backward,
)


def sqrt_backward(result, a, grad_output):
    """
    c = sqrt(a)
    dc/da = 1/(2 * sqrt(a)) * grad_output
    """
    return (grad_output / (2 * np.sqrt(a.value)),)


sqrt_op = Operation(
    name="sqrt",
    forward_func=lambda a: a.sqrt(),
    backward_func=sqrt_backward,
)
