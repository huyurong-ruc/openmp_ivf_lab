import csv
import statistics as st
from pathlib import Path

# 设置实验E结果路径
result_dir = Path("results/experiment_E")
raw_dir = result_dir / "raw"

threads_list = [1, 16]
distance_modes = ["scalar", "simd"]
strategies = [
    ("omp1", "static"),
    ("omp1", "dynamic,1"),
    ("omp1", "guided,1"),
    ("omp2", "not_applicable"),
]

results = {}

# 读取16组正式计时结果
for strategy, schedule in strategies:
    schedule_name = schedule.replace(",", "_")
    for threads in threads_list:
        for distance in distance_modes:
            csv_path = raw_dir / (
                f"{strategy}_{schedule_name}_t{threads}_{distance}.csv"
            )

            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            first = rows[0]
            times = [float(row["time_ms"]) for row in rows]
            recalls = [float(row["recall"]) for row in rows]

            results[(strategy, schedule, threads, distance)] = {
                "strategy": first["strategy"],
                "schedule": first["omp_schedule"],
                "distance": first["distance"],
                "threads": int(first["threads"]),
                "batch": int(first["batch"]),
                "queries": int(first["queries"]),
                "nprobe": int(first["nprobe"]),
                "k": int(first["k"]),
                "grain": int(first["grain"]),
                "min_ms": min(times),
                "median_ms": st.median(times),
                "max_ms": max(times),
                "mean_ms": st.mean(times),
                "recall": st.median(recalls),
                "correctness": "PASS" if all(
                    row["correctness"] == "PASS" for row in rows
                ) else "FAIL",
            }

summary = []

# 计算相同策略和线程数下SIMD相对Scalar的加速比
for strategy, schedule in strategies:
    for threads in threads_list:
        scalar = results[(strategy, schedule, threads, "scalar")]
        simd = results[(strategy, schedule, threads, "simd")]

        for field in ("threads", "batch", "queries", "nprobe", "k", "grain"):
            assert scalar[field] == simd[field]

        speedup = scalar["median_ms"] / simd["median_ms"]

        summary.append({
            "strategy": strategy,
            "schedule": schedule,
            "threads": threads,
            "batch": scalar["batch"],
            "queries": scalar["queries"],
            "nprobe": scalar["nprobe"],
            "k": scalar["k"],
            "grain": scalar["grain"],
            "scalar_min_ms": scalar["min_ms"],
            "scalar_median_ms": scalar["median_ms"],
            "scalar_max_ms": scalar["max_ms"],
            "scalar_mean_ms": scalar["mean_ms"],
            "simd_min_ms": simd["min_ms"],
            "simd_median_ms": simd["median_ms"],
            "simd_max_ms": simd["max_ms"],
            "simd_mean_ms": simd["mean_ms"],
            "simd_speedup": speedup,
            "scalar_recall": scalar["recall"],
            "simd_recall": simd["recall"],
            "scalar_correctness": scalar["correctness"],
            "simd_correctness": simd["correctness"],
        })

        print(
            f"{strategy} {schedule} threads={threads}: "
            f"{scalar['median_ms']:.3f} -> {simd['median_ms']:.3f} ms | "
            f"speedup={speedup:.3f} | "
            f"{scalar['correctness']}/{simd['correctness']}"
        )

# 保存实验E汇总结果，每个策略与线程数组合一行
with open(
    result_dir / "summary.csv", "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"分析完成：{result_dir / 'summary.csv'}")
