import csv
import statistics as st
from pathlib import Path
import numpy as np

# 设置数据和结果路径
data_dir = Path("data")
result_dir = Path("results/experiment_B")
raw_dir = result_dir / "raw"

Q = 128
NPROBE = 8
NLIST = 4096
DIM = 128

# 读取聚类中心，每行包含维度标记和128个float32坐标
centroid_data = np.fromfile(data_dir / "centroid_4096.fvecs", dtype="<i4")
centroids = centroid_data.reshape(NLIST, DIM + 1)[:, 1:].copy().view("<f4")

# 读取前128个查询向量
with open(data_dir / "sift_query.fbin", "rb") as f:
    rows, dim = np.fromfile(f, dtype="<u4", count=2)
    queries = np.fromfile(f, dtype="<f4", count=Q * DIM).reshape(Q, DIM)

# 读取数据库向量所属的聚类编号
cluster_data = np.fromfile(data_dir / "cluster_id_4096.ivecs", dtype="<i4")
cluster_ids = cluster_data.reshape(-1, 2)[:, 1]

# 统计4096个倒排列表的长度
list_lengths = np.bincount(cluster_ids, minlength=NLIST)

# 计算每个查询需要扫描的候选数C(q)
candidate_counts = []

for query in queries:
    # 使用double计算到各聚类中心的平方欧氏距离
    delta = centroids.astype(np.float64) - query.astype(np.float64)
    distances = np.sum(delta * delta, axis=1)

    # 按距离升序排列，等距时按聚类编号升序排列
    selected = np.lexsort((np.arange(NLIST), distances))[:NPROBE]

    # 对选中的8个倒排列表长度求和
    candidate_counts.append(int(list_lengths[selected].sum()))

# 保存列表长度和查询工作量，供绘图使用
with open(result_dir / "list_lengths.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["list_id", "length"])
    writer.writerows(enumerate(list_lengths))

with open(result_dir / "candidate_counts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "candidate_count"])
    writer.writerows(enumerate(candidate_counts))

# 汇总查询工作量统计
list_min = int(min(list_lengths))
list_median = float(st.median(list_lengths))
list_max = int(max(list_lengths))

candidate_min = min(candidate_counts)
candidate_median = float(st.median(candidate_counts))
candidate_max = max(candidate_counts)
candidate_total = sum(candidate_counts)

print(f"列表长度：min={list_min}, median={list_median}, max={list_max}")
print(
    f"C(q)：min={candidate_min}, median={candidate_median}, "
    f"max={candidate_max}, total={candidate_total}"
)

# 读取8组正式计时CSV
results = {}

for file in sorted(raw_dir.glob("*.csv")):
    with open(file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    first = rows[0]
    key = (first["strategy"], first["omp_schedule"], first["query_order"])
    times = [float(row["time_ms"]) for row in rows]

    results[key] = {
        "min_ms": min(times),
        "median_ms": st.median(times),
        "max_ms": max(times),
        "mean_ms": st.mean(times),
        "qps": int(first["queries"]) * 1000 / st.median(times),
        "correctness": "PASS" if all(
            row["correctness"] == "PASS" for row in rows
        ) else "FAIL",
        "recall": float(first["recall"]),
    }

# 比较每种策略在原始顺序和排序后的运行时间
strategies = [
    ("omp1", "static"),
    ("omp1", "dynamic,1"),
    ("omp1", "guided,1"),
    ("omp2", "not_applicable"),
]

summary = []

for strategy, schedule in strategies:
    original = results[(strategy, schedule, "original")]
    sorted_result = results[(strategy, schedule, "sorted")]

    # 正值表示排序后变慢，负值表示排序后变快
    change = (
        sorted_result["median_ms"] / original["median_ms"] - 1
    ) * 100

    summary.append({
        "strategy": strategy,
        "schedule": schedule,
        "original_min_ms": original["min_ms"],
        "original_median_ms": original["median_ms"],
        "original_max_ms": original["max_ms"],
        "original_mean_ms": original["mean_ms"],
        "sorted_min_ms": sorted_result["min_ms"],
        "sorted_median_ms": sorted_result["median_ms"],
        "sorted_max_ms": sorted_result["max_ms"],
        "sorted_mean_ms": sorted_result["mean_ms"],
        "original_qps": original["qps"],
        "sorted_qps": sorted_result["qps"],
        "change_percent": change,
        "original_correctness": original["correctness"],
        "sorted_correctness": sorted_result["correctness"],
        "original_recall": original["recall"],
        "sorted_recall": sorted_result["recall"],
        "list_min": list_min,
        "list_median": list_median,
        "list_max": list_max,
        "candidate_min": candidate_min,
        "candidate_median": candidate_median,
        "candidate_max": candidate_max,
        "candidate_total": candidate_total,
    })

    print(
        f"{strategy} {schedule}: "
        f"{original['median_ms']:.3f} -> "
        f"{sorted_result['median_ms']:.3f} ms "
        f"({change:+.2f}%)"
    )

# 保存实验B汇总表
with open(result_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"分析完成：{result_dir / 'summary.csv'}")
