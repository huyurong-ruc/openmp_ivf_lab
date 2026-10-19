#!/usr/bin/env bash
set -euo pipefail

# 设置OpenMP环境变量，与实验B保持一致
export OMP_WAIT_POLICY=PASSIVE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset OMP_PLACES

# 设置结果保存位置
results_dir="results/experiment_C/raw"
mkdir -p "$results_dir"

# 运行4组不同grain的OMP2配置
for grain in 64 256 1024 4096; do
    csv="$results_dir/omp2_grain_${grain}.csv"

    env -u OMP_SCHEDULE ./ivf_lab \
        --base data/sift_base.fbin \
        --centroids data/centroid_4096.fvecs \
        --cluster-ids data/cluster_id_4096.ivecs \
        --queries-file data/sift_query.fbin \
        --gt data/gt.ibin \
        --only omp2 \
        --query-order original \
        --queries 128 \
        --batch 64 \
        --k 10 \
        --nprobe 8 \
        --threads 16 \
        --grain "$grain" \
        --warmup 1 \
        --repeats 10 \
        --csv "$csv"
done
