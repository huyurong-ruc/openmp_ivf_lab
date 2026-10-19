#!/usr/bin/env bash
set -euo pipefail

# 设置OpenMP环境变量
export OMP_WAIT_POLICY=PASSIVE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset OMP_PLACES

# 设置结果保存位置
results_dir="results/experiment_D/raw"
mkdir -p "$results_dir"

# 运行一组配置
run_config() {
    local strategy=$1
    local schedule=$2
    local nprobe=$3
    local schedule_name=${schedule//,/_}
    local csv="$results_dir/${strategy}_${schedule_name}_np${nprobe}.csv"
    local -a command=(
        ./ivf_lab
        --base data/sift_base.fbin
        --centroids data/centroid_4096.fvecs
        --cluster-ids data/cluster_id_4096.ivecs
        --queries-file data/sift_query.fbin
        --gt data/gt.ibin
        --only "$strategy"
        --query-order original
        --queries 128
        --batch 64
        --k 10
        --nprobe "$nprobe"
        --threads 16
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

# 遍历3种搜索范围和4种并行策略
for nprobe in 1 8 32; do
    for schedule in static dynamic,1 guided,1; do
        run_config omp1 "$schedule" "$nprobe"
    done
    run_config omp2 not_applicable "$nprobe"
done
