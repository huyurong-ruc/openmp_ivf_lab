import csv
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

# 读取实验D汇总结果
result_dir = Path("results/experiment_D")

with open(result_dir / "summary.csv", newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

nprobes = [1, 8, 32]

# 同一nprobe的候选数和Recall不随并行策略改变，只绘制一组数据
mean_candidates = []
recalls = []

for nprobe in nprobes:
    rows = [row for row in summary if int(row["nprobe"]) == nprobe]
    mean_candidates.append(float(rows[0]["mean_candidates"]))
    recalls.append(st.median(float(row["recall"]) for row in rows))

# 图1：Recall与平均候选数随nprobe的变化
fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

axes[0].plot(nprobes, mean_candidates, marker="o")
axes[0].set_ylabel("Mean Candidate Count")
axes[0].set_title("Recall and Candidate Count vs. nprobe")
axes[0].grid(axis="y", alpha=0.3)

axes[1].plot(nprobes, recalls, marker="o")
axes[1].set_xlabel("nprobe")
axes[1].set_ylabel("Recall@10")
axes[1].set_xticks(nprobes)
axes[1].grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(result_dir / "recall_candidate_count.png", dpi=300)
plt.close(fig)


# 图2：4种并行策略的Recall-QPS权衡
series = [
    ("omp1", "static", "OMP1 static"),
    ("omp1", "dynamic,1", "OMP1 dynamic,1"),
    ("omp1", "guided,1", "OMP1 guided,1"),
    ("omp2", "not_applicable", "OMP2 task"),
]

plt.figure(figsize=(7, 4.5))

for strategy, schedule, label in series:
    rows = [
        row for row in summary
        if row["strategy"] == strategy and row["schedule"] == schedule
    ]
    rows.sort(key=lambda row: int(row["nprobe"]))

    x = [float(row["recall"]) for row in rows]
    y = [float(row["qps"]) for row in rows]

    plt.plot(x, y, marker="o", label=label)

rows = [
    row for row in summary
    if row["strategy"] == "omp1" and row["schedule"] == "dynamic,1"
]
rows.sort(key=lambda row: int(row["nprobe"]))

for row in rows:
    plt.annotate(
        f"np={row['nprobe']}",
        (float(row["recall"]), float(row["qps"])),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

plt.xlabel("Recall@10")
plt.ylabel("QPS")
plt.title("Recall-QPS Trade-off")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(result_dir / "recall_qps_tradeoff.png", dpi=300)
plt.close()

print("已生成实验D的2张图。")