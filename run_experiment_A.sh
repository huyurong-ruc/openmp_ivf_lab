#!/usr/bin/env bash
set -euo pipefail

# 设置OpenMP环境变量
export OMP_WAIT_POLICY=PASSIVE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset OMP_PLACES

# 设置结果保存位置
results_dir="results/experiment_A/raw"
mkdir -p "$results_dir"

# 运行一组配置
run_config() {
    local strategy=$1
    local schedule=$2
    local threads=$3
    local batch=$4
    local schedule_name=${schedule//,/_}
    local csv="$results_dir/${strategy}_${schedule_name}_t${threads}_b${batch}.csv"
    local -a command=(
        ./ivf_lab
        --base data/sift_base.fbin
        --centroids data/centroid_4096.fvecs
        --cluster-ids data/cluster_id_4096.ivecs
        --queries-file data/sift_query.fbin
        --gt data/gt.ibin
        --only "$strategy"
        --queries 128
        --batch "$batch"
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
}

# 运行两组串行基线
for batch in 1 64; do
    run_config serial not_applicable 1 "$batch"
done

# 遍历不同配置的batch、threads、schedule
for batch in 1 64; do
    for threads in 1 2 4 8 16; do
        for schedule in static dynamic,1 guided,1; do
            run_config omp1 "$schedule" "$threads" "$batch"
        done
        run_config omp2 not_applicable "$threads" "$batch"
    done
done