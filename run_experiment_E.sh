#!/usr/bin/env bash
set -euo pipefail

# 设置OpenMP环境变量
export OMP_WAIT_POLICY=PASSIVE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset OMP_PLACES

# 设置结果保存位置
results_dir="results/experiment_E/raw"
mkdir -p "$results_dir"

# 运行一组配置
run_config() {
    local strategy=$1
    local schedule=$2
    local threads=$3
    local distance=$4
    local schedule_name=${schedule//,/_}
    local csv="$results_dir/${strategy}_${schedule_name}_t${threads}_${distance}.csv"
    local -a command=(
        ./ivf_lab
        --base data/sift_base.fbin
        --centroids data/centroid_4096.fvecs
        --cluster-ids data/cluster_id_4096.ivecs
        --queries-file data/sift_query.fbin
        --gt data/gt.ibin
        --only "$strategy"
        --distance "$distance"
        --query-order original
        --queries 128
        --batch 64
        --k 10
        --nprobe 8
        --threads "$threads"
        --grain 1024
        --warmup 1
        --repeats 10
        --csv "$csv"
    )

    if [[ $schedule == "not_applicable" ]]; then
        env -u OMP_SCHEDULE "${command[@]}"
    else
        OMP_SCHEDULE="$schedule" "${command[@]}"
    fi

    sed -i '1s/$/,omp_schedule/' "$csv"
    sed -i "2,\$s/\$/,\"$schedule\"/" "$csv"
    sed -i '1s/$/,distance/' "$csv"
    sed -i "2,\$s/\$/,\"$distance\"/" "$csv"
}

# 遍历线程数和距离计算模式
for threads in 1 16; do
    for distance in scalar simd; do
        for schedule in static dynamic,1 guided,1; do
            run_config omp1 "$schedule" "$threads" "$distance"
        done
        run_config omp2 not_applicable "$threads" "$distance"
    done
done
