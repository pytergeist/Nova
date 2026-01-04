# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import numpy as np
import pytest

from nova.src.initialisers import RandomNormal, RandomSeed, RandomUniform

"""
TODO: Implement pytest requests for grabbing the initialiser class for generic tests
"""


def test_random_seed_initialiser_init_seed():
    random_seed = RandomSeed(seed=42)
    assert random_seed.seed == 42


def test_random_seed_initialiser_default_seed():
    random_seed = RandomSeed()
    assert isinstance(random_seed.seed, int)


def test_random_seed_initialiser_get_config():
    random_seed = RandomSeed(seed=42)
    assert random_seed.get_config() == {"seed": 42}


@pytest.mark.parametrize(
    "mean, expected",
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
    ],
)
def test_random_normal_initialiser_init_mean(mean, expected):
    random_normal = RandomNormal(mean=mean)
    assert random_normal.mean == expected


@pytest.mark.parametrize(
    "stddev, expected",
    [
        (1.0, 1.0),
        (2.0, 2.0),
        (0.5, 0.5),
    ],
)
def test_random_normal_initialiser_init_stddev(stddev, expected):
    random_normal = RandomNormal(stddev=stddev)
    assert random_normal.stddev == expected


def test_random_normal_initialiser_get_config():
    random_normal = RandomNormal(mean=1.0, stddev=2.0, seed=42)
    assert random_normal.get_config() == {"seed": 42, "mean": 1.0, "stddev": 2.0}


def test_random_normal_generate_random_normal_data_shape():
    random_normal = RandomNormal(mean=0.0, stddev=1.0, seed=42)
    data = random_normal._generate_random_normal_data((4, 4))
    assert data.shape == (4, 4)


@pytest.mark.parametrize(
    "seed, shape, mean, stddev",
    [
        (42, (4, 4), 0.0, 1.0),
        (42, (4, 4, 4), 0.0, 1.0),
        (42, (4, 4, 4, 4), 0.0, 1.0),
        (42, (4, 4, 4, 4, 4), 0.0, 1.0),
    ],
)
def test_random_normal_generate_random_normal_data_values(seed, shape, mean, stddev):
    # This test is here for when a custom implementation of the _generate_random_normal_data method is built
    rng = np.random.default_rng(seed)
    expected = rng.normal(loc=mean, scale=stddev, size=shape)
    random_normal = RandomNormal(mean=mean, stddev=stddev, seed=seed)
    data = random_normal._generate_random_normal_data(shape)
    np.testing.assert_array_equal(data, expected)


# def test_random_normal_call(): # TODO: Implement shape/dtype property for core, need to fix node issue first
#     random_normal = RandomNormal(mean=0.0, stddev=1.0, seed=42)
#     core = random_normal((4, 4), dtype="float32")
#     assert core.shape == (4, 4)
#     assert core.dtype == "float32"


@pytest.mark.parametrize(
    "minval, expected",
    [
        (-1.0, -1.0),
        (0.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_random_uniform_initialiser_init_minval(minval, expected):
    random_uniform = RandomUniform(minval=minval)
    assert random_uniform.minval == expected


@pytest.mark.parametrize(
    "maxval, expected",
    [
        (-1.0, -1.0),
        (0.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_random_uniform_initialiser_init_maxval(maxval, expected):
    random_uniform = RandomUniform(maxval=maxval)
    assert random_uniform.maxval == expected


def test_ensure_minval_less_than_maxval():
    random_uniform = RandomUniform(minval=1.0, maxval=-1.0)
    with pytest.raises(ValueError):
        random_uniform._ensure_minval_less_than_maxval()


def test_random_uniform_initialiser_get_config():
    random_uniform = RandomUniform(minval=-1.0, maxval=1.0, seed=42)
    assert random_uniform.get_config() == {"seed": 42, "minval": -1.0, "maxval": 1.0}


def test_random_uniform_generate_random_uniform_data_shape():
    random_uniform = RandomUniform(minval=-1.0, maxval=1.0, seed=42)
    data = random_uniform._generate_randon_uniform_data((4, 4))
    assert data.shape == (4, 4)


def test_random_uniform_generate_random_uniform_data_values():
    random_uniform = RandomUniform(minval=-1.0, maxval=1.0, seed=42)
    data = random_uniform._generate_randon_uniform_data((4, 4))
    assert np.all(data >= -1.0) and np.all(data <= 1.0)


# def test_random_uniform_call(): # TODO: Implement shape/dtype property for core, need to fix node issue first
#     random_uniform = RandomUniform(minval=-1.0, maxval=1.0, seed=42)
#     core = random_uniform((4, 4), dtype="float32")
#     assert core.shape == (4, 4)
#     assert core.dtype == "float32"


if __name__ == "__main__":
    pytest.main([__file__])
