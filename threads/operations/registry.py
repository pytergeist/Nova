# registry.py
from .operation import Operation


########################
# ADD
########################
def add_backward(result, a, b, grad_output):
    """
    c = a + b
    derivative wrt a = grad_output
    derivative wrt b = grad_output
    """
    return (grad_output, grad_output)


add_op = Operation("add", lambda a, b: a.data + b.data, add_backward)


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


subtract_op = Operation("subtract", lambda a, b: a.data - b.data, subtract_backward)


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
    "right_subtract", lambda a, b: b.data - a.data, right_subtract_backward
)


########################
# DIVIDE
########################
def divide_backward(result, a, b, grad_output):
    grad_a = grad_output / b.value
    grad_b = -grad_output * a.value / (b.value**2)
    return (grad_a, grad_b)


divide_op = Operation("divide", lambda a, b: a.data / b.data, divide_backward)
