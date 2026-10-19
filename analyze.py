import csv
import statistics as st
from pathlib import Path

# 设置输入输出路径
result_dir = Path("results/experiment_A")
raw_dir = result_dir / "raw"
summary_file = result_dir / "summary.csv"

# 读取42组原始数据
results = []
for file in sorted(raw_dir.glob("*.csv")):
    with open(file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    first = rows[0]
    times = [float(row["time_ms"]) for row in rows]
    queries = int(first["queries"])

    # 计算10次运行的时间统计
    result = {
        "strategy": first["strategy"],
        "schedule": first["omp_schedule"],
        "threads": int(first["threads"]),
        "batch": int(first["batch"]),
        "queries": queries,
        "k": int(first["k"]),
        "nprobe": int(first["nprobe"]),
        "grain": int(first["grain"]),
        "min_ms": min(times),
        "median_ms": st.median(times),
        "max_ms": max(times),
        "mean_ms": st.mean(times),
        "qps_mean": st.mean(queries * 1000 / t for t in times),
        "correctness": "PASS" if all(row["correctness"] == "PASS" for row in rows) else "FAIL",
        "recall": float(first["recall"]),
    }

    results.append(result)

# 获取不同batch下的串行基线
serial_times = {
    row["batch"]: row["median_ms"]
    for row in results
    if row["strategy"] == "serial"
}

# 计算加速比和并行效率
for row in results:
    baseline = serial_times[row["batch"]]
    row["speedup"] = baseline / row["median_ms"]
    row["efficiency"] = row["speedup"] / row["threads"]

# 保存汇总结果
fields = [
    "strategy", "schedule", "threads", "batch",
    "queries", "k", "nprobe", "grain",
    "min_ms", "median_ms", "max_ms", "mean_ms",
    "qps_mean", "speedup", "efficiency",
    "correctness", "recall",
]

with open(summary_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"完成：{len(results)}组配置，结果保存到 {summary_file}")
