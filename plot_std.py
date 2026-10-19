import csv
from pathlib import Path

import matplotlib.pyplot as plt

# 读取std::thread对照实验汇总结果
result_dir = Path("results/experiment_std")

with open(result_dir / "summary.csv", newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

threads = [1, 2, 4, 8, 16]
serial = next(row for row in summary if row["strategy"] == "serial")
serial_time = float(serial["median_ms"])

series = [
    ("omp1", "static", "OMP1 static"),
    ("std", "not_applicable", "std::thread"),
]

# 图1：OMP1与std::thread的中位运行时间
plt.figure(figsize=(7, 4.5))

for strategy, schedule, label in series:
    rows = sorted(
        [
            row for row in summary
            if row["strategy"] == strategy and row["schedule"] == schedule
        ],
        key=lambda row: int(row["threads"]),
    )
    times = [float(row["median_ms"]) for row in rows]
    plt.plot(threads, times, marker="o", label=label)

plt.axhline(serial_time, linestyle="--", linewidth=1, label="Serial")
plt.xticks(threads)
plt.xlabel("Number of Threads")
plt.ylabel("Median Time (ms)")
plt.title("OMP1 vs. std::thread Search Time")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(result_dir / "std_time.png", dpi=300)
plt.close()

# 图2：OMP1与std::thread相对串行版本的加速比
plt.figure(figsize=(7, 4.5))

for strategy, schedule, label in series:
    rows = sorted(
        [
            row for row in summary
            if row["strategy"] == strategy and row["schedule"] == schedule
        ],
        key=lambda row: int(row["threads"]),
    )
    speedups = [float(row["speedup"]) for row in rows]
    plt.plot(threads, speedups, marker="o", label=label)

plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xticks(threads)
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup Relative to Serial")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(result_dir / "std_speedup.png", dpi=300)
plt.close()

print("已生成std::thread对照实验的2张图。")
