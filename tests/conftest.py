import pytest
import numpy as np


@pytest.fixture
def fn_numpy():
    def _fn_numpy(x, fn_str):
        fn = getattr(np, fn_str)
        return fn(x, x)

    return _fn_numpy
