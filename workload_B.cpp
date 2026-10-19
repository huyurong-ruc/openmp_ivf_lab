#include "ivf_search.h"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t Q = 128;
constexpr int THREADS = 16;

struct Chunk {
    std::size_t begin;
    std::size_t end;
};

// 计算一个查询需要扫描的候选数C(q)
std::size_t candidate_count(const ivf::IVFIndex& index,
                            const float* query, std::size_t nprobe) {
    const auto lists = ivf::select_lists(index, query, nprobe);
    std::size_t count = 0;

    for (auto list : lists) {
        count += index.list_ptr[list + 1] - index.list_ptr[list];
    }
    return count;
}

// OMP1：按查询并行，统计每个线程实际扫描的候选数
std::vector<std::size_t> workload_omp1(const ivf::IVFIndex& index, const std::vector<float>& queries, const ivf::SearchOptions& options, omp_sched_t schedule, int chunk_size) {
    std::vector<std::size_t> work(THREADS, 0);
    std::vector<ivf::Neighbor> output(Q * options.k);

    omp_set_schedule(schedule, chunk_size);

    // 与student1保持相同的查询级并行方式
    for (std::size_t begin = 0; begin < Q; begin += 64) {
        const std::size_t count = std::min<std::size_t>(64, Q - begin);

        #pragma omp parallel for num_threads(THREADS) schedule(runtime)
        for (std::size_t q = 0; q < count; ++q) {
            const float* query = queries.data() + (begin + q) * index.dim;
            const auto lists = ivf::select_lists(index, query, options.nprobe);
            ivf::TopK best(options.k);

            std::size_t scanned = 0;

            // 保留完整候选扫描与Top-k计算
            for (auto list : lists) {
                for (std::size_t p = index.list_ptr[list];
                     p < index.list_ptr[list + 1]; ++p) {
                    best.add({
                        ivf::l2_distance_scalar(query, index.vector(p), index.dim),
                        index.ids[p]
                    });
                    ++scanned;
                }
            }

            best.finish(output.data() + (begin + q) * options.k);

            // 每个线程只修改自己的计数器
            const int tid = omp_get_thread_num();
            work[tid] += scanned;
        }
    }

    return work;
}

// OMP2：按候选块生成task，统计执行task的线程工作量
std::vector<std::size_t> workload_omp2(
    const ivf::IVFIndex& index,
    const std::vector<float>& queries,
    const ivf::SearchOptions& options
) {
    std::vector<std::size_t> work(THREADS, 0);
    std::vector<ivf::Neighbor> output(Q * options.k);

    // 与student2保持相同的task并行结构
    for (std::size_t begin = 0; begin < Q; begin += 64) {
        const std::size_t count = std::min<std::size_t>(64, Q - begin);

        #pragma omp parallel num_threads(THREADS)
        {
            #pragma omp single
            {
                for (std::size_t q = 0; q < count; ++q) {
                    const float* query = queries.data() + (begin + q) * index.dim;
                    const auto lists = ivf::select_lists(index, query, options.nprobe);

                    // 按grain拆分候选，长列表也可以拆成多个task
                    std::vector<Chunk> chunks;

                    for (auto list : lists) {
                        const std::size_t end = index.list_ptr[list + 1];

                        for (std::size_t p = index.list_ptr[list];
                             p < end; p += options.grain) {
                            chunks.push_back({
                                p, std::min(p + options.grain, end)
                            });
                        }
                    }

                    // 每个task使用独立的局部Top-k
                    std::vector<std::vector<ivf::Neighbor>>
                        local_results(chunks.size());

                    #pragma omp taskgroup
                    {
                        for (std::size_t i = 0; i < chunks.size(); ++i) {
                            #pragma omp task firstprivate(i, query) shared(chunks, local_results, index, options, work)
                            {
                                ivf::TopK local_best(options.k);

                                for (std::size_t p = chunks[i].begin;
                                     p < chunks[i].end; ++p) {
                                    local_best.add({
                                        ivf::l2_distance_scalar(
                                            query, index.vector(p), index.dim
                                        ),
                                        index.ids[p]
                                    });
                                }

                                local_results[i].resize(options.k);
                                local_best.finish(local_results[i].data());

                                // task执行完后，计入实际执行它的线程
                                const int tid = omp_get_thread_num();
                                work[tid] += chunks[i].end - chunks[i].begin;
                            }
                        }
                    }

                    // 等待所有task完成后合并局部Top-k
                    ivf::TopK best(options.k);

                    for (const auto& local : local_results) {
                        for (const auto& neighbor : local) {
                            if (neighbor.id != -1) {
                                best.add(neighbor);
                            }
                        }
                    }

                    best.finish(output.data() + (begin + q) * options.k);
                }
            }
        }
    }

    return work;
}

// 保存一组配置的线程工作量
void save_workload(
    std::ofstream& csv,
    const std::string& strategy,
    const std::string& schedule,
    const std::string& order,
    const std::vector<std::size_t>& work,
    std::size_t expected_total
) {
    const std::size_t total =
        std::accumulate(work.begin(), work.end(), std::size_t{0});

    // 检查所有线程的候选计数之和是否正确
    if (total != expected_total) {
        throw std::runtime_error("workload count mismatch");
    }

    const std::size_t max_work =
        *std::max_element(work.begin(), work.end());

    const double mean_work = double(total) / THREADS;
    const double imbalance = max_work / mean_work;

    csv << strategy << ',' << schedule << ',' << order;

    for (auto value : work) {
        csv << ',' << value;
    }

    csv << ',' << total << ',' << imbalance << '\n';

    std::cout << strategy << " " << schedule << " " << order
              << "  total=" << total
              << "  imbalance=" << imbalance << '\n';
}

} // namespace

int main() {
    // 读取与实验B相同的SIFT1M数据和索引
    const auto index = ivf::read_ivf(
        "data/sift_base.fbin",
        "data/centroid_4096.fvecs",
        "data/cluster_id_4096.ivecs"
    );
    const auto query_matrix = ivf::read_fbin("data/sift_query.fbin");

    ivf::SearchOptions options;
    options.k = 10;
    options.nprobe = 8;
    options.threads = THREADS;
    options.grain = 1024;

    omp_set_dynamic(0);

    // 计算原始128个查询的候选数C(q)
    std::vector<std::size_t> counts(Q);
    std::vector<std::size_t> query_order(Q);

    std::iota(query_order.begin(), query_order.end(), 0);

    for (std::size_t q = 0; q < Q; ++q) {
        counts[q] = candidate_count(
            index, query_matrix.row(q), options.nprobe
        );
    }

    const std::size_t total_candidates =
        std::accumulate(counts.begin(), counts.end(), std::size_t{0});

    std::ofstream csv("results/experiment_B/workload.csv");

    // 每行保存16个线程的实际候选计算量
    csv << "strategy,schedule,query_order";

    for (int t = 0; t < THREADS; ++t) {
        csv << ",W" << t;
    }

    csv << ",total_candidates,imbalance_ratio\n";

    // 比较原始顺序与按C(q)升序排序后的顺序
    for (const std::string order : {"original", "sorted"}) {
        if (order == "sorted") {
            std::sort(query_order.begin(), query_order.end(),
                [&](std::size_t a, std::size_t b) {
                    if (counts[a] != counts[b]) {
                        return counts[a] < counts[b];
                    }
                    return a < b;
                });
        }

        // 按当前顺序连续存储查询，与正式实验B保持一致
        std::vector<float> queries(Q * index.dim);

        for (std::size_t q = 0; q < Q; ++q) {
            const float* source = query_matrix.row(query_order[q]);

            std::copy_n(
                source, index.dim,
                queries.data() + q * index.dim
            );
        }

        // OMP1分别使用三种调度方式
        const auto static_work = workload_omp1(
            index, queries, options, omp_sched_static, 0
        );
        save_workload(csv, "omp1", "static", order,
                      static_work, total_candidates);

        const auto dynamic_work = workload_omp1(
            index, queries, options, omp_sched_dynamic, 1
        );
        save_workload(csv, "omp1", "dynamic,1", order,
                      dynamic_work, total_candidates);

        const auto guided_work = workload_omp1(
            index, queries, options, omp_sched_guided, 1
        );
        save_workload(csv, "omp1", "guided,1", order,
                      guided_work, total_candidates);

        // OMP2采用相同的grain=1024
        const auto task_work = workload_omp2(index, queries, options);
        save_workload(csv, "omp2", "not_applicable", order,
                      task_work, total_candidates);
    }

    std::cout << "Saved: results/experiment_B/workload.csv\n";
    return 0;
}
