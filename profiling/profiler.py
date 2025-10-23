import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    """Tiny wall-clock profiler: use with profiler.phase('name')."""

    def __init__(self):
        self.stats = defaultdict(list)
        self._stack = []

    @contextmanager
    def phase(self, name):
        t0 = time.perf_counter_ns()
        self._stack.append(name)
        try:
            yield
        finally:
            dt = (time.perf_counter_ns() - t0) / 1e6  # ms
            self.stats[name].append(dt)
            self._stack.pop()

    def summary(self, header=None):
        if header:
            print(header)
        totals = {k: sum(v) for k, v in self.stats.items()}
        total_all = sum(totals.values()) or 1.0
        lines = []
        for k, tot in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            arr = self.stats[k]
            lines.append(
                f"  {k:>12}: total {tot:9.2f} ms | mean {sum(arr)/len(arr):7.3f} ms "
                f"| median {sorted(arr)[len(arr)//2]:7.3f} ms | calls {len(arr):5d} "
                f"| {100*tot/total_all:5.1f}%"
            )
        print("\n".join(lines))
        print(f"  {'TOTAL':>12}: {total_all:9.2f} ms\n")

    def reset(self):
        self.stats.clear()
