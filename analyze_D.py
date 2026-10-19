import csv
import statistics as st
from pathlib import Path

import numpy as np

# 设置实验D的数据和结果路径
data_dir = Path("data")
result_dir = Path("results/experiment_D")
raw_dir = result_dir / "raw"

Q = 128
DIM = 128
NLIST = 4096
NPROBES = [1, 8, 32]

strategies = [
    ("omp1", "static"),
    ("omp1", "dynamic,1"),
    ("omp1", "guided,1"),
    ("omp2", "not_applicable"),
]

# 读取聚类中心，每行包含维度标记和128个float32坐标
centroid_data = np.fromfile(
    data_dir / "centroid_4096.fvecs", dtype="<i4"
).reshape(NLIST, DIM + 1)

centroids = (
    centroid_data[:, 1:].copy()
    .view("<f4")
    .reshape(NLIST, DIM)
    .astype(np.float64)
)

# 读取前128个查询向量
with open(data_dir / "sift_query.fbin", "rb") as f:
    rows, dim = np.fromfile(f, dtype="<u4", count=2)
    queries = np.fromfile(
        f, dtype="<f4", count=Q * DIM
    ).reshape(Q, DIM)

# 读取数据库向量所属的聚类编号并统计列表长度
cluster_data = np.fromfile(
    data_dir / "cluster_id_4096.ivecs", dtype="<i4"
).reshape(-1, 2)

cluster_ids = cluster_data[:, 1]
list_lengths = np.bincount(cluster_ids, minlength=NLIST)

# 按C++标量距离口径计算每个查询的中心次序
center_orders = []

for query in queries:
    delta = centroids - query.astype(np.float64)
    distances = np.sum(delta * delta, axis=1)
    center_orders.append(np.lexsort((np.arange(NLIST), distances)))

candidate_stats = {}

for nprobe in NPROBES:
    candidate_counts = [
        int(list_lengths[order[:nprobe]].sum())
        for order in center_orders
    ]
    candidate_stats[nprobe] = {
        "total_candidates": sum(candidate_counts),
        "mean_candidates": st.mean(candidate_counts),
        "median_candidates": st.median(candidate_counts),
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
    }

summary = []

# 汇总12组正式计时结果，并复用同一nprobe的候选数统计
for nprobe in NPROBES:
    for strategy, schedule in strategies:
        schedule_name = schedule.replace(",", "_")
        csv_path = raw_dir / f"{strategy}_{schedule_name}_np{nprobe}.csv"

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        first = rows[0]
        times = [float(row["time_ms"]) for row in rows]
        recalls = [float(row["recall"]) for row in rows]
        median_ms = st.median(times)

        summary.append({
            "strategy": first["strategy"],
            "schedule": first["omp_schedule"],
            "nprobe": int(first["nprobe"]),
            "threads": int(first["threads"]),
            "batch": int(first["batch"]),
            "queries": int(first["queries"]),
            "k": int(first["k"]),
            "grain": int(first["grain"]),
            **candidate_stats[nprobe],
            "min_ms": min(times),
            "median_ms": median_ms,
            "max_ms": max(times),
            "mean_ms": st.mean(times),
            "qps": int(first["queries"]) * 1000 / median_ms,
            "recall": st.median(recalls),
            "correctness": "PASS" if all(
                row["correctness"] == "PASS" for row in rows
            ) else "FAIL",
        })

        print(
            f"{strategy} {schedule} nprobe={nprobe}: "
            f"median={median_ms:.3f} ms | "
            f"qps={summary[-1]['qps']:.2f} | "
            f"recall={summary[-1]['recall']:.6f} | "
            f"{summary[-1]['correctness']}"
        )

# 保存实验D汇总结果，每组配置一行
with open(
    result_dir / "summary.csv", "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"分析完成：{result_dir / 'summary.csv'}")
