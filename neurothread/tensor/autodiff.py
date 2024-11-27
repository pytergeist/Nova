# autodiff.py
import numpy as np


class AutoDiff:

    @staticmethod
    def add_backward(tensor, other, grad_output):
        with tensor.lock:
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = grad_output
                else:
                    tensor.grad += grad_output
        with other.lock:
            if other.requires_grad:
                if other.grad is None:
                    other.grad = grad_output
                else:
                    other.grad += grad_output

    @staticmethod
    def subtract_backward(tensor, other, grad_output):
        with tensor.lock:
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = grad_output
                else:
                    tensor.grad += grad_output
        with other.lock:
            if other.requires_grad:
                if other.grad is None:
                    other.grad = -grad_output
                else:
                    other.grad -= grad_output
