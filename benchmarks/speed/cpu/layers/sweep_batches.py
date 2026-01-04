# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import math
from typing import List

import matplotlib.pyplot as plt

from linear_bench import profile_linear


def main():
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    common_kwargs = dict(
        in_features=1024,
        out_features=1024,
        dtype="float32",
        warmup=10,
        inner=50,
        samples=15,
        threads=1,
        seed=42,
    )

    results = []
    for b in batch_sizes:
        print(f"\n=== Running benchmark for batch={b} ===")
        res = profile_linear(batch=b, verbose=False, **common_kwargs)
        results.append(res)

    batches = [r["batch"] for r in results]

    nova_model_med = [r["nova_model_med"] for r in results]
    nova_raw_med = [r["nova_raw_med"] for r in results]
    torch_med = [r["torch_med"] for r in results]
    keras_med = [r["keras_med"] for r in results]

    def to_speed(ts):
        return [1.0 / t if t > 0 else math.nan for t in ts]

    nova_model_speed = to_speed(nova_model_med)
    nova_raw_speed = to_speed(nova_raw_med)
    torch_speed = to_speed(torch_med)
    keras_speed = to_speed(keras_med)

    # --- Fig 1: nova_model vs Torch vs Keras ---
    plt.figure()
    plt.plot(batches, nova_model_speed, marker="o", label="Nova (model)")
    plt.plot(batches, torch_speed, marker="o", label="PyTorch")
    plt.plot(batches, keras_speed, marker="o", label="Keras")
    plt.xscale("log", base=2)  # optional: log-scale for clearer spread
    plt.xlabel("Batch size")
    plt.ylabel("Speed (runs / second)")
    plt.title("Linear layer speed vs batch size (Nova model vs PyTorch vs Keras)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)

    # --- Fig 2: nova_raw vs Torch vs Keras ---
    plt.figure()
    plt.plot(batches, nova_raw_speed, marker="o", label="Nova (raw)")
    plt.plot(batches, torch_speed, marker="o", label="PyTorch")
    plt.plot(batches, keras_speed, marker="o", label="Keras")
    plt.xscale("log", base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Speed (runs / second)")
    plt.title("Linear kernel speed vs batch size (Nova raw vs PyTorch vs Keras)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
