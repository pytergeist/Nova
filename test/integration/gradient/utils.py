import functools

from nova.src.backend.core.clib import grad_tape


def set_grad_tape(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with grad_tape():
            return func(*args, **kwargs)

    return wrapper
