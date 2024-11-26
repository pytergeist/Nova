# autograd.py
import numpy as np


class Autograd:

    @staticmethod
    def add_backward(tensor, other, grad_output):
        with tensor.lock:
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = np.ones_like(tensor.data) * grad_output
                else:
                    tensor.grad += np.ones_like(tensor.data) * grad_output
        with other.lock:
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.ones_like(other.data) * grad_output
                else:
                    other.grad += np.ones_like(other.data) * grad_output

    @staticmethod
    def subtract_backward(tensor, other, grad_output):
        with tensor.lock:
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = np.ones_like(tensor.data) * grad_output
                else:
                    tensor.grad += np.ones_like(tensor.data) * grad_output
        with other.lock:
            if other.requires_grad:
                if other.grad is None:
                    other.grad = -np.ones_like(other.data) * grad_output
                else:
                    other.grad -= np.ones_like(other.data) * grad_output
