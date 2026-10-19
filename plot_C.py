import csv
from pathlib import Path

import matplotlib.pyplot as plt

result_dir = Path("results/experiment_C")

with open(result_dir / "summary.csv", newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

summary.sort(key=lambda row: int(row["grain"]))

grains = [int(row["grain"]) for row in summary]
labels = [str(grain) for grain in grains]

times = [float(row["median_ms"]) for row in summary]
min_times = [float(row["min_ms"]) for row in summary]
max_times = [float(row["max_ms"]) for row in summary]

tasks = [int(row["total_tasks"]) for row in summary]
memory = [float(row["peak_local_kib"]) for row in summary]

x = list(range(len(grains)))

# 图1：任务粒度与运行时间
lower_errors = [
    median - minimum
    for median, minimum in zip(times, min_times)
]
upper_errors = [
    maximum - median
    for median, maximum in zip(times, max_times)
]

plt.figure(figsize=(7, 4.5))
plt.errorbar(
    x,
    times,
    yerr=[lower_errors, upper_errors],
    marker="o",
    capsize=5,
)
plt.xticks(x, labels)
plt.xlabel("Grain")
plt.ylabel("Median Time (ms)")
plt.title("Effect of Task Grain on OMP2 Performance")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    result_dir / "grain_time.png", dpi=300
)
plt.close()

# 图2：任务粒度与总task数量
plt.figure(figsize=(7, 4.5))
plt.plot(x, tasks, marker="o")
plt.xticks(x, labels)
plt.xlabel("Grain")
plt.ylabel("Total Number of Tasks")
plt.title("Task Count vs. Grain")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    result_dir / "grain_task_count.png", dpi=300
)
plt.close()

# 图3：任务粒度与局部Top-k结果的峰值估算空间
plt.figure(figsize=(7, 4.5))
plt.plot(x, memory, marker="o")
plt.xticks(x, labels)
plt.xlabel("Grain")
plt.ylabel("Estimated Peak Local Top-k Storage (KiB)")
plt.title("Local Top-k Storage vs. Grain")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    result_dir / "grain_temp_memory.png", dpi=300
)
plt.close()

print("已生成实验C的3张图。")
