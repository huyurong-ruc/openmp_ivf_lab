import csv
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

# 读取实验A汇总数据
result_dir = Path("results/experiment_A")

with open(result_dir / "summary.csv", newline="", encoding="utf-8") as f:
    results = list(csv.DictReader(f))

strategies = [
    ("omp1", "static", "OMP1 static"),
    ("omp1", "dynamic,1", "OMP1 dynamic,1"),
    ("omp1", "guided,1", "OMP1 guided,1"),
    ("omp2", "not_applicable", "OMP2 task"),
]

# 分别绘制两种batch下的加速比和并行效率
for batch in [1, 64]:
    for metric, ylabel in [
        ("speedup", "Speedup"),
        ("efficiency", "Parallel Efficiency"),
    ]:
        plt.figure(figsize=(7, 4.5))

        for strategy, schedule, label in strategies:
            data = sorted(
                [
                    row for row in results
                    if row["strategy"] == strategy
                    and row["schedule"] == schedule
                    and int(row["batch"]) == batch
                ],
                key=lambda row: int(row["threads"])
            )

            threads = [int(row["threads"]) for row in data]
            values = [float(row[metric]) for row in data]

            plt.plot(threads, values, marker="o", label=label)

        # 加速比为1表示与串行持平
        if metric == "speedup":
            plt.axhline(1, linestyle="--", linewidth=1)

        plt.xticks([1, 2, 4, 8, 16])
        plt.xlabel("Number of Threads")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Threads (batch={batch})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # 每张图单独保存
        filename = f"{metric}_batch{batch}.png"
        plt.savefig(result_dir / filename, dpi=300)
        plt.close()

# 绘制batch=1的单查询延迟p50/p95
raw_dir = result_dir / "raw"

latency_strategies = [
    ("serial", "not_applicable", "Serial"),
    ("omp1", "static", "OMP1 static"),
    ("omp1", "dynamic,1", "OMP1 dynamic,1"),
    ("omp1", "guided,1", "OMP1 guided,1"),
    ("omp2", "not_applicable", "OMP2 task"),
]

plt.figure(figsize=(8, 5))

for strategy, schedule, label in latency_strategies:
    data = []

    for file in raw_dir.glob("*.csv"):
        with open(file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        first = rows[0]
        if (
            first["strategy"] != strategy
            or first["omp_schedule"] != schedule
            or int(first["batch"]) != 1
        ):
            continue

        threads = int(first["threads"])
        p50 = statistics.median(float(row["p50_ms"]) for row in rows)
        p95 = statistics.median(float(row["p95_ms"]) for row in rows)
        data.append((threads, p50, p95))

    if not data:
        continue

    data.sort()
    threads, p50_values, p95_values = zip(*data)

    plt.plot(threads, p50_values, marker="o", label=f"{label} p50")
    plt.plot(threads, p95_values, marker="x", linestyle="--", label=f"{label} p95")

plt.xticks([1, 2, 4, 8, 16])
plt.xlabel("Number of Threads")
plt.ylabel("Latency (ms)")
plt.title("Single-query Latency (batch=1)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(result_dir / "latency_batch1.png", dpi=300)
plt.close()

print("已生成5张实验A图表")