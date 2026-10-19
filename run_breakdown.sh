#!/usr/bin/env bash
set -euo pipefail

# 设置与检索实验相同的OpenMP环境
export OMP_WAIT_POLICY=PASSIVE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset OMP_PLACES
unset OMP_SCHEDULE

results_dir="results/breakdown"
mkdir -p "$results_dir"
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT

# 使用项目编译选项构建并运行诊断程序
g++ -I. -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp -pthread \
    breakdown_omp2.cpp ivf_io.cpp ivf_search.cpp ivf_student.cpp \
    -o "$build_dir/breakdown_omp2"
"$build_dir/breakdown_omp2"
