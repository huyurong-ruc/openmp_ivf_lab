#!/usr/bin/env bash
set -euo pipefail

# 提权后再创建通道，避免sudo关闭继承的FD
if (( EUID != 0 )); then
    printf '%s\n' '请在项目根目录执行：sudo bash run_perf.sh' >&2
    exit 1
fi

export LC_ALL=C
export OMP_WAIT_POLICY=PASSIVE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
unset OMP_PLACES
unset OMP_SCHEDULE

results_dir="results/perf"
mkdir -p "$results_dir"
channel_dir=$(mktemp -d)
trap 'rm -rf -- "$channel_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cpu_set=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
test -n "$cpu_set"

# 记录共同CPU集合和运行环境
{
    perf --version
    uname -sr
    id
    printf 'perf_event_paranoid='; cat /proc/sys/kernel/perf_event_paranoid
    printf 'allowed_cpus=%s\n' "$cpu_set"
    taskset -c "$cpu_set" sh -c 'grep Cpus_allowed_list /proc/self/status'
    env | sort | grep '^OMP_'
    printf '%s\n' 'thread_binding=none; all processes inherit the same CPU set'
    printf '%s\n' 'distance=scalar query_order=original queries=128 batch=64 k=10 nprobe=8 grain=1024 warmup=1 repeats=10'
    printf '%s\n' 'formal_windows=20 query_instances=1280; default perf inheritance enabled'
    printf '%s\n' 'task-clock is cumulative CPU time; perf time elapsed is not search time'
    printf '%s\n' 'control-boundary overhead may be counted; ack waits are outside benchmark time_ms'
} > "$results_dir/environment.txt"

# 只接受三个完整软件事件的有效数值，不接受受限用户态事件
check_counts() {
    awk -F ';' '
        $3 == "task-clock" || $3 == "context-switches" || $3 == "cpu-migrations" {
            value=$1; gsub(/^[ \t]+|[ \t]+$/, "", value)
            if (value !~ /^[0-9]+([.][0-9]+)?$/) bad=1
            if ($3 == "task-clock" && value+0 <= 0) bad=1
            seen[$3]++
        }
        END {exit (bad || seen["task-clock"] != 1 || seen["context-switches"] != 1 || seen["cpu-migrations"] != 1)}
    ' "$1"
}

# 在加载数据前确认采集权限
perf stat -x ';' --no-big-num -o "$channel_dir/probe.txt" \
    -e task-clock,context-switches,cpu-migrations -- true
check_counts "$channel_dir/probe.txt"

# 每组复用一个perf进程，累计正式batch窗口的计数
run_config() {
    local name=$1 strategy=$2 threads=$3 schedule=$4
    local control_fd ack_fd
    local csv="$results_dir/${name}.csv.pending"
    local counters="$results_dir/${name}_perf.txt.pending"
    mkfifo "$channel_dir/${name}.control" "$channel_dir/${name}.ack"
    exec {control_fd}<>"$channel_dir/${name}.control"
    exec {ack_fd}<>"$channel_dir/${name}.ack"
    local -a environment=(env -u OMP_SCHEDULE)
    if [[ $schedule == static ]]; then
        environment+=(OMP_SCHEDULE=static)
    fi
    taskset -c "$cpu_set" perf stat --delay=-1 \
        --control="fd:$control_fd,$ack_fd" \
        -e task-clock,context-switches,cpu-migrations \
        -x ';' --no-big-num -o "$counters" -- \
        "${environment[@]}" ./ivf_lab \
        --base data/sift_base.fbin \
        --centroids data/centroid_4096.fvecs \
        --cluster-ids data/cluster_id_4096.ivecs \
        --queries-file data/sift_query.fbin \
        --gt data/gt.ibin \
        --only "$strategy" --threads "$threads" \
        --distance scalar --query-order original \
        --queries 128 --batch 64 --k 10 --nprobe 8 --grain 1024 \
        --warmup 1 --repeats 10 --csv "$csv" \
        --perf-control-fd "$control_fd" --perf-ack-fd "$ack_fd"
    exec {control_fd}>&-
    exec {ack_fd}>&-
    check_counts "$counters"
    awk -F ',' 'NR>1 {if ($12 != "PASS") bad=1} END {exit (bad || NR != 11)}' "$csv"
    # 校验成功后发布结果，失败时保留pending文件供排查
    mv -- "$counters" "$results_dir/${name}_perf.txt"
    mv -- "$csv" "$results_dir/${name}.csv"
    printf '%s strategy=%s threads=%s OMP_SCHEDULE=%s\n' \
        "$name" "$strategy" "$threads" "$schedule" >> "$results_dir/environment.txt"
}

run_config serial serial 1 not_applicable
run_config omp1_static omp1 16 static
run_config std std 16 not_applicable
run_config omp2_task omp2 16 not_applicable
