# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import numpy as np
import pytest

from nova.src.backend.core import Tensor
from nova.src.backend.topology.builder import Builder
from nova.src.blocks.activations import ReLU
from tests.integration.gradient import set_grad
from tests.integration.gradient.finite_difference import (
    Tolerance,
    finite_difference_jacobian,
)

# TODO: Add parameterisation for multiple test cases
# TODO: Implement integration tests for larger computation graphs


def fn_numpy(x):
    y = x + x
    return np.maximum(0, y)


@set_grad
def compute_autodiff_gradient(x):
    x_tensor = Tensor(x, requires_grad=True)

    tensor_output = x_tensor + x_tensor

    with Builder():  # TODO: change | temporary fix for builder context
        output = ReLU().call(tensor_output)

    output.backward()

    return x_tensor.grad.to_numpy()


def test_addition_grad():
    x_test = np.random.uniform(-1, 1, 5)

    numerical_jacobian = finite_difference_jacobian(
        fn_numpy, x_test, epsilon=Tolerance.EPSILON.value
    )
    numerical_vector_grad = np.dot(np.ones(x_test.shape), numerical_jacobian)
    analytical_grad = compute_autodiff_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=Tolerance.RTOL.value,
        atol=Tolerance.ATOL.value,
        err_msg="Autodiff gradient does not match numerical gradient for addition.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
