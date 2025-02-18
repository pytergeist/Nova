import numpy as np


def finite_difference_jacobian(f, x, epsilon=1.5e-8):
    """
    Compute an approximate Jacobian matrix using the finite difference method.

    This function generalizes the finite difference method for approximating the
    Jacobian matrix using the formula:

        f'(x_i) = (f(x + ε * e_i) - f(x - ε * e_i)) / (2 * ε)

    It can be used to compute both Jacobians and scalar gradients.

    Args:
        x (scalar or array-like): A scalar or vector of shape (n,).
        f (callable): A transformation function that takes an input of shape (n,)
            and returns an output of shape (m,).
        epsilon (float, optional): A small scalar value for the finite difference
            approximation. Defaults to 1.5e-8.

    Returns:
        numpy.ndarray: The approximate Jacobian matrix of first-order partial derivatives,
            with shape (m, n).
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    f0 = np.atleast_1d(np.asarray(f(x), dtype=np.float64)).ravel()
    jacobian = np.zeros((f0.shape[0], x.size), dtype=np.float64)
    for i in range(x.size):
        dx = np.zeros_like(x, dtype=np.float32)
        dx[i] = epsilon

        f_pos = np.atleast_1d(np.array(f(x + dx), dtype=np.float64)).ravel()
        f_neg = np.atleast_1d(np.array(f(x - dx), dtype=np.float64)).ravel()

        jacobian[:, i] = (f_pos - f_neg) / (2 * epsilon)

        if jacobian.size == 1:
            jacobian = jacobian.ravel()

    return jacobian
