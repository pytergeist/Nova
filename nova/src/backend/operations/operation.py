# operation.py


class Operation:
    def __init__(self, name, forward_func, backward_func):
        """Represents an operation with forward and backward logic.

        Args:`x     op_name (str): Name of the operation (e.g., "add").     forward_func
        (callable): Function that computes the forward pass.     backward_func
        (callable): Function that computes the backward pass.
        """
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func
