#include "ivf_search.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Times {
    double total = 0;
    double prepare = 0;
    double task = 0;
    double merge = 0;
    double other = 0;
};

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

// 保持原有任务划分和合并过程，记录single线程看到的阶段墙钟时间
void search_omp2_breakdown(const ivf::IVFIndex& index, const float* queries,
                           std::size_t count, const ivf::SearchOptions& options,
                           ivf::Neighbor* output, Times& times) {
    #pragma omp parallel num_threads(options.threads)
    {
        #pragma omp single
        {
            for (std::size_t q = 0; q < count; q++) {
                const float* query = queries + q * index.dim;
                const auto distance = options.use_simd ? ivf::l2_distance_simd
                                                       : ivf::l2_distance_scalar;

                const auto prepare_start = Clock::now();
                const auto lists = ivf::select_lists(index, query, options.nprobe);

                struct Chunk {
                    std::size_t begin;
                    std::size_t end;
                };
                std::vector<Chunk> chunks;

                for (auto list : lists) {
                    const std::size_t end = index.list_ptr[list + 1];
                    for (std::size_t p = index.list_ptr[list]; p < end; p += options.grain) {
                        chunks.push_back({p, std::min(p + options.grain, end)});
                    }
                }

                std::vector<std::vector<ivf::Neighbor>> local_results(chunks.size());
                times.prepare += milliseconds(prepare_start, Clock::now());

                // 包含任务创建、候选计算及taskgroup等待
                const auto task_start = Clock::now();
                #pragma omp taskgroup
                {
                    for (std::size_t i = 0; i < chunks.size(); i++) {
                        #pragma omp task firstprivate(i, query, distance) shared(chunks, local_results, index, options)
                        {
                            ivf::TopK local_best(options.k);
                            for (std::size_t p = chunks[i].begin; p < chunks[i].end; p++) {
                                local_best.add({
                                    distance(query, index.vector(p), index.dim),
                                    index.ids[p]
                                });
                            }

                            local_results[i].resize(options.k);
                            local_best.finish(local_results[i].data());
                        }
                    }
                }
                times.task += milliseconds(task_start, Clock::now());

                // 合并局部Top-k并写出查询结果
                const auto merge_start = Clock::now();
                ivf::TopK best(options.k);
                for (const auto& local : local_results) {
                    for (const auto& neighbor : local) {
                        if (neighbor.id != -1) {
                            best.add(neighbor);
                        }
                    }
                }

                best.finish(output + q * options.k);
                times.merge += milliseconds(merge_start, Clock::now());
            }
        }
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

const char* environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "unset";
}

int run() {
    constexpr std::size_t query_count = 128, batch = 64, repeats = 10, warmup = 1;
    ivf::SearchOptions options;
    options.k = 10;
    options.nprobe = 8;
    options.threads = 16;
    options.grain = 1024;
    options.use_simd = false;

    const auto index = ivf::read_ivf("data/sift_base.fbin", "data/centroid_4096.fvecs",
                                     "data/cluster_id_4096.ivecs");
    const auto queries = ivf::read_fbin("data/sift_query.fbin");
    const auto truth = ivf::read_ibin("data/gt.ibin");
    if (queries.dim != index.dim || query_count > queries.rows || query_count > truth.rows ||
        options.k > index.size || options.k > truth.dim || options.nprobe > index.nlist) {
        throw std::invalid_argument("query count, dimension, k or nprobe exceeds input bounds");
    }
    for (std::size_t q = 0; q < query_count; ++q) {
        const auto* row = truth.row(q);
        for (std::size_t j = 0; j < options.k; ++j) {
            if (row[j] < 0 || std::size_t(row[j]) >= index.size ||
                std::find(row, row + j, row[j]) != row + j) {
                throw std::invalid_argument("invalid or duplicate ground truth ID");
            }
        }
    }

    omp_set_dynamic(0);
    omp_set_num_threads(options.threads);
    std::vector<ivf::Neighbor> reference(query_count * options.k), output(reference.size());
    ivf::search_ivf_serial(index, queries.data.data(), query_count, options, reference.data());
    const double recall = ivf::recall_at_k(reference, truth, query_count, options.k);

    auto run_once = [&] {
        std::fill(output.begin(), output.end(),
                  ivf::Neighbor{std::numeric_limits<double>::quiet_NaN(), -2});
        Times times;
        for (std::size_t begin = 0; begin < query_count; begin += batch) {
            const auto count = std::min(batch, query_count - begin);
            const auto start = Clock::now();
            search_omp2_breakdown(index, queries.row(begin), count, options,
                                  output.data() + begin * options.k, times);
            times.total += milliseconds(start, Clock::now());
        }
        ivf::check_result(output, reference);
        times.other = times.total - times.prepare - times.task - times.merge;
        if (times.other < -0.01) throw std::runtime_error("invalid negative other_ms");
        return times;
    };

    for (std::size_t r = 0; r < warmup; ++r) run_once();
    std::vector<Times> runs;
    runs.reserve(repeats);
    for (std::size_t r = 0; r < repeats; ++r) runs.push_back(run_once());

    // 全部正式轮次通过检查后再写入统计文件
    std::ofstream csv("results/breakdown/breakdown_omp2.csv");
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << "repeat,total_ms,prepare_ms,task_ms,merge_ms,other_ms,"
           "prepare_pct,task_pct,merge_pct,other_pct,correctness,recall\n"
        << std::setprecision(12);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto& t = runs[r];
        csv << r + 1 << ',' << t.total << ',' << t.prepare << ',' << t.task << ','
            << t.merge << ',' << t.other << ',' << 100 * t.prepare / t.total << ','
            << 100 * t.task / t.total << ',' << 100 * t.merge / t.total << ','
            << 100 * t.other / t.total << ",PASS," << recall << '\n';
    }
    csv.close();

    std::ofstream summary("results/breakdown/summary.txt");
    summary.exceptions(std::ios::failbit | std::ios::badbit);
    summary << std::fixed << std::setprecision(6)
            << "strategy=omp2 threads=16 queries=128 batch=64 k=10 nprobe=8 grain=1024 "
               "distance=scalar query_order=original warmup=1 repeats=10\n"
            << "compiler=g++ flags=-I. -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp -pthread\n"
            << "OMP_WAIT_POLICY=" << environment("OMP_WAIT_POLICY")
            << " OMP_DYNAMIC=" << environment("OMP_DYNAMIC")
            << " OMP_PROC_BIND=" << environment("OMP_PROC_BIND")
            << " OMP_PLACES=" << environment("OMP_PLACES")
            << " OMP_SCHEDULE=" << environment("OMP_SCHEDULE") << '\n'
            << "timing=steady_clock wall time; two synchronous batch calls per repeat; "
               "phase times are not sums of worker CPU times\n"
            << "percent=100 * sum(phase_ms) / sum(total_ms) over 10 repeats; "
               "medians are computed independently by column\n"
            << "phase min_ms median_ms max_ms percent\n";
    const auto report = [&](const char* name, double Times::*member) {
        std::vector<double> values;
        double sum = 0, total = 0;
        for (const auto& t : runs) {
            values.push_back(t.*member);
            sum += t.*member;
            total += t.total;
        }
        summary << name << ' ' << *std::min_element(values.begin(), values.end()) << ' '
                << median(values) << ' ' << *std::max_element(values.begin(), values.end())
                << ' ' << 100 * sum / total << '\n';
    };
    report("total", &Times::total);
    report("prepare", &Times::prepare);
    report("task", &Times::task);
    report("merge", &Times::merge);
    report("other", &Times::other);
    summary << "correctness=PASS recall@10=" << std::setprecision(9) << recall << '\n';
    std::cout << "correctness=PASS recall@10=" << std::fixed << std::setprecision(9)
              << recall << "\n";
    return 0;
}

}

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
