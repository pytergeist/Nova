# Autodiff.py


class AutoDiff:
    BACKWARD_FUNCS = {}

    @staticmethod
    def register(op_name, backward_func):
        """
        Register a backward function for a given operation name.

        Args:
            op_name (str): The name of the operation.
            backward_func (function): A function that takes one argument

        Returns:
            None
        """
        AutoDiff.BACKWARD_FUNCS[op_name] = backward_func

    @staticmethod
    def apply(op_name, *args, **kwargs):
        """
        Applies the backward function for a given operation name.

        Args:
            ops_name (str): The name of the operation.
            *args: Arguments for the backward function.

        Returns:
            None
        """
        if op_name not in AutoDiff.BACKWARD_FUNCS:
            raise ValueError(f"No backward function for operation {op_name}")
        AutoDiff.BACKWARD_FUNCS[op_name](*args, **kwargs)

    @staticmethod
    def _generic_backward_func(tensor, grad_output):
        with tensor.lock:
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = grad_output
                else:
                    tensor.grad += grad_output


# Register backward functions
AutoDiff.register(
    "add",
    lambda tensor, other, grad_output: (
        AutoDiff._generic_backward_func(tensor, grad_output),
        AutoDiff._generic_backward_func(other, grad_output),
    ),
)


AutoDiff.register(
    "subtract",
    lambda tensor, other, grad_output: (
        AutoDiff._generic_backward_func(tensor, grad_output),
        AutoDiff._generic_backward_func(other, -grad_output),
    ),
)
