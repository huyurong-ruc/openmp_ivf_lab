import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 读取实验B分析结果
result_dir = Path("results/experiment_B")

with open(result_dir / "summary.csv", newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

with open(result_dir / "list_lengths.csv", newline="", encoding="utf-8") as f:
    list_lengths = [int(row["length"]) for row in csv.DictReader(f)]

with open(result_dir / "candidate_counts.csv", newline="", encoding="utf-8") as f:
    candidate_counts = [
        int(row["candidate_count"]) for row in csv.DictReader(f)
    ]

# 图1：4096个倒排列表的长度分布
plt.figure(figsize=(7, 4.5))
plt.hist(list_lengths, bins=40, edgecolor="white")
plt.xlabel("List Length")
plt.ylabel("Number of Lists")
plt.title("IVF Inverted List Length Distribution")
plt.tight_layout()
plt.savefig(result_dir / "list_length_distribution.png", dpi=300)
plt.close()

# 图2：128个查询的候选数C(q)分布
plt.figure(figsize=(7, 4.5))
plt.hist(candidate_counts, bins=25, edgecolor="white")
plt.xlabel("Candidate Count C(q)")
plt.ylabel("Number of Queries")
plt.title("Query Candidate Count Distribution")
plt.tight_layout()
plt.savefig(result_dir / "candidate_count_distribution.png", dpi=300)
plt.close()

# 图3：原始顺序与排序后的中位运行时间
labels = [
    f"{row['strategy']} {row['schedule']}"
    for row in summary
]

original_times = [float(row["original_median_ms"]) for row in summary]
sorted_times = [float(row["sorted_median_ms"]) for row in summary]

x = np.arange(len(summary))
width = 0.35

plt.figure(figsize=(8, 4.5))
plt.bar(x - width / 2, original_times, width, label="Original")
plt.bar(x + width / 2, sorted_times, width, label="Sorted")

plt.xticks(x, labels)
plt.ylabel("Median Time (ms)")
plt.title("Effect of Query Ordering")
plt.legend()
plt.tight_layout()
plt.savefig(result_dir / "query_order_comparison.png", dpi=300)
plt.close()


# 图4：比较排序前后各并行策略的线程负载不均衡比
workload = {}

with open(result_dir / "workload.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头

    for row in reader:
        strategy = row[0]

        # 兼容workload.csv中未加引号的dynamic,1和guided,1
        if row[1] in ("dynamic", "guided"):
            schedule = row[1] + "," + row[2]
            order = row[3]
        else:
            schedule = row[1]
            order = row[2]

        workload[(strategy, schedule, order)] = float(row[-1])

strategies = [
    ("omp1", "static"),
    ("omp1", "dynamic,1"),
    ("omp1", "guided,1"),
    ("omp2", "not_applicable"),
]

labels = ["OMP1 static", "OMP1 dynamic", "OMP1 guided", "OMP2 task"]

original = [
    workload[(strategy, schedule, "original")]
    for strategy, schedule in strategies
]
sorted_values = [
    workload[(strategy, schedule, "sorted")]
    for strategy, schedule in strategies
]

x = np.arange(len(strategies))
width = 0.35

plt.figure(figsize=(8, 4.5))
plt.bar(x - width / 2, original, width, label="Original")
plt.bar(x + width / 2, sorted_values, width, label="Sorted")
plt.axhline(1.0, linestyle="--", linewidth=1, label="Perfect Balance")

plt.xticks(x, labels)
plt.ylabel("Load Imbalance Ratio")
plt.title("Thread Workload Imbalance")
plt.legend()
plt.tight_layout()
plt.savefig(result_dir / "workload_imbalance_comparison.png", dpi=300)
plt.close()

print("已生成实验B的4张图")
