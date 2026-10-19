
import csv
import math
import statistics as st
from pathlib import Path

import numpy as np

# 实验C固定参数
data_dir = Path("data")
result_dir = Path("results/experiment_C")
raw_dir = result_dir / "raw"

Q = 128
DIM = 128
NLIST = 4096
NPROBE = 8
K = 10
GRAINS = [64, 256, 1024, 4096]

# Neighbor包含double距离和int32编号，当前64位G++环境下含对齐填充，共16字节
NEIGHBOR_BYTES = 16

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

# 读取数据库向量所属的聚类编号
cluster_data = np.fromfile(
    data_dir / "cluster_id_4096.ivecs", dtype="<i4"
).reshape(-1, 2)

cluster_ids = cluster_data[:, 1]

# 统计4096个倒排列表的长度
list_lengths = np.bincount(cluster_ids, minlength=NLIST)

# 计算每个查询选中的8个倒排列表
selected_lists = []

for query in queries:
    # 使用double计算到各聚类中心的平方欧氏距离
    delta = centroids - query.astype(np.float64)
    distances = np.sum(delta * delta, axis=1)

    # 按距离升序排列，距离相同时按聚类编号升序
    selected = np.lexsort((np.arange(NLIST), distances))[:NPROBE]
    selected_lists.append(selected)

# 计算全部查询需要扫描的候选向量数
total_candidates = sum(
    int(list_lengths[selected].sum())
    for selected in selected_lists
)

summary = []

# 依次分析4种grain配置
for grain in GRAINS:
    # 每个倒排列表独立切分task，不能直接用C(q)/grain
    tasks_per_query = []

    for selected in selected_lists:
        task_count = sum(
            math.ceil(int(list_lengths[list_id]) / grain)
            for list_id in selected
        )
        tasks_per_query.append(task_count)

    total_tasks = sum(tasks_per_query)
    max_tasks = max(tasks_per_query)

    # 每个task最多保留K个局部Top-k结果槽位
    total_local_slots = total_tasks * K
    peak_local_slots = max_tasks * K

    # 估算局部Top-k结果的峰值空间
    peak_local_bytes = peak_local_slots * NEIGHBOR_BYTES
    peak_local_kib = peak_local_bytes / 1024

    # 读取该grain配置的10次正式计时
    csv_path = raw_dir / f"omp2_grain_{grain}.csv"

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    times = [float(row["time_ms"]) for row in rows]
    recalls = [float(row["recall"]) for row in rows]

    correctness = (
        "PASS"
        if all(row["correctness"] == "PASS" for row in rows)
        else "FAIL"
    )

    # 汇总时间、任务数和临时空间
    summary.append({
        "grain": grain,
        "threads": 16,
        "batch": 64,
        "queries": Q,
        "nprobe": NPROBE,
        "k": K,
        "total_candidates": total_candidates,
        "total_tasks": total_tasks,
        "min_tasks_per_query": min(tasks_per_query),
        "median_tasks_per_query": st.median(tasks_per_query),
        "max_tasks_per_query": max_tasks,
        "total_local_slots": total_local_slots,
        "peak_local_slots": peak_local_slots,
        "peak_local_bytes": peak_local_bytes,
        "peak_local_kib": peak_local_kib,
        "min_ms": min(times),
        "median_ms": st.median(times),
        "max_ms": max(times),
        "mean_ms": st.mean(times),
        "recall": st.median(recalls),
        "correctness": correctness,
    })

    print(
        f"grain={grain:4d} | "
        f"tasks={total_tasks:5d} | "
        f"peak={peak_local_kib:.2f} KiB | "
        f"median={st.median(times):.3f} ms | "
        f"{correctness}"
    )

# 保存实验C汇总结果，每个grain一行
with open(
    result_dir / "summary.csv", "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"\n分析完成：{result_dir / 'summary.csv'}")
