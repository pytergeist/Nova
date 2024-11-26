def add(tensor_a, tensor_b):
    """
    Element-wise addition of two tensors.
    """
    data = tensor_a.data + tensor_b.data
    requires_grad = tensor_a.requires_grad or tensor_b.requires_grad
    return data, requires_grad


def subtract(tensor_a, tensor_b):
    """
    Element-wise subtraction of two tensors.
    """
    data = tensor_a.data - tensor_b.data
    requires_grad = tensor_a.requires_grad or tensor_b.requires_grad
    return data, requires_grad
