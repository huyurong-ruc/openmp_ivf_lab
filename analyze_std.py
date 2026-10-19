import csv
import statistics as st
from pathlib import Path

# 设置std::thread对照实验结果路径
result_dir = Path("results/experiment_std")
raw_dir = result_dir / "raw"

threads_list = [1, 2, 4, 8, 16]
configurations = [("serial", "not_applicable", 1)]

for threads in threads_list:
    configurations.append(("omp1", "static", threads))
    configurations.append(("std", "not_applicable", threads))

results = []

# 读取11组正式计时结果
for strategy, schedule, threads in configurations:
    schedule_name = schedule.replace(",", "_")
    csv_path = raw_dir / f"{strategy}_{schedule_name}_t{threads}.csv"

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    first = rows[0]
    times = [float(row["time_ms"]) for row in rows]
    recalls = [float(row["recall"]) for row in rows]

    results.append({
        "strategy": first["strategy"],
        "schedule": first["omp_schedule"],
        "threads": int(first["threads"]),
        "batch": int(first["batch"]),
        "queries": int(first["queries"]),
        "k": int(first["k"]),
        "nprobe": int(first["nprobe"]),
        "grain": int(first["grain"]),
        "distance": first["distance"],
        "min_ms": min(times),
        "median_ms": st.median(times),
        "max_ms": max(times),
        "mean_ms": st.mean(times),
        "correctness": "PASS" if all(
            row["correctness"] == "PASS" for row in rows
        ) else "FAIL",
        "recall": st.median(recalls),
    })

serial = next(row for row in results if row["strategy"] == "serial")

# 使用串行中位时间计算加速比
for row in results:
    for field in ("batch", "queries", "k", "nprobe", "grain", "distance"):
        assert row[field] == serial[field]
    row["speedup"] = serial["median_ms"] / row["median_ms"]

    print(
        f"{row['strategy']} {row['schedule']} threads={row['threads']}: "
        f"median={row['median_ms']:.3f} ms | "
        f"speedup={row['speedup']:.3f} | {row['correctness']}"
    )

# 保存每组配置的统计结果
fields = [
    "strategy", "schedule", "threads", "batch", "queries", "k",
    "nprobe", "grain", "distance", "min_ms", "median_ms", "max_ms",
    "mean_ms", "speedup", "correctness", "recall",
]

with open(
    result_dir / "summary.csv", "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"分析完成：{result_dir / 'summary.csv'}")
