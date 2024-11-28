# operation.py
from neurothread.autodiff.autodiff import AutoDiff


class Operation:
    def __init__(self, op_name, forward_func, backward_func):
        """
        Represents an operation with forward and backward logic.

        Args:
            op_name (str): Name of the operation (e.g., "add").
            forward_func (callable): Function that computes the forward pass.
            backward_func (callable): Function that computes the backward pass.
        """
        self.op_name = op_name
        self.forward_func = forward_func
        self.backward_func = backward_func
        AutoDiff.register(op_name, backward_func)
