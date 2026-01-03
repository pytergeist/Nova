import functools

from nova.src.backend.core import Grad


def set_grad(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Grad():
            return func(*args, **kwargs)

    return wrapper
