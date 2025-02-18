import numpy as np


def finite_difference_jacobian(f, x, epsilon=1.5e-8):
    """
    fn to generalise finite difference method for approximating the
    jacobian matrix using the below equation:
    f'(x_i) = (f(x + ε*e_i) - f(x - ε*e_i)) / 2*ε)
    This fn can compute both jacobian and scalar gradients
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
