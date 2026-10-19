import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 读取实验E汇总结果
result_dir = Path("results/experiment_E")

with open(result_dir / "summary.csv", newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

series = [
    ("omp1", "static", "OMP1 static"),
    ("omp1", "dynamic,1", "OMP1 dynamic,1"),
    ("omp1", "guided,1", "OMP1 guided,1"),
    ("omp2", "not_applicable", "OMP2 task"),
]

labels = [label for _, _, label in series]
x = np.arange(len(series))
width = 0.35

# 图1：分别比较1线程和16线程下的Scalar与SIMD时间
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for axis, threads in zip(axes, (1, 16)):
    rows = []
    for strategy, schedule, _ in series:
        rows.append(next(
            row for row in summary
            if row["strategy"] == strategy
            and row["schedule"] == schedule
            and int(row["threads"]) == threads
        ))

    scalar_times = [float(row["scalar_median_ms"]) for row in rows]
    simd_times = [float(row["simd_median_ms"]) for row in rows]

    axis.bar(x - width / 2, scalar_times, width, label="Scalar")
    axis.bar(x + width / 2, simd_times, width, label="SIMD")
    axis.set_xticks(x, labels, rotation=15)
    axis.set_ylabel("Median Time (ms)")
    axis.set_title(f"Threads = {threads}")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

fig.suptitle("Scalar vs. SIMD Search Time")
fig.tight_layout()
fig.savefig(result_dir / "scalar_simd_time.png", dpi=300)
plt.close(fig)

# 图2：比较1线程和16线程下的SIMD加速比
speedups = {}

for threads in (1, 16):
    speedups[threads] = []
    for strategy, schedule, _ in series:
        row = next(
            row for row in summary
            if row["strategy"] == strategy
            and row["schedule"] == schedule
            and int(row["threads"]) == threads
        )
        speedups[threads].append(float(row["simd_speedup"]))

plt.figure(figsize=(9, 4.5))
plt.bar(x - width / 2, speedups[1], width, label="Threads = 1")
plt.bar(x + width / 2, speedups[16], width, label="Threads = 16")
plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
plt.xticks(x, labels, rotation=15)
plt.ylabel("SIMD Speedup")
plt.title("SIMD Speedup by Strategy and Thread Count")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(result_dir / "simd_speedup.png", dpi=300)
plt.close()

print("已生成实验E的2张图。")
