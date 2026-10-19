#include "ivf_search.h"
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <thread>

namespace ivf {

// C++标准线程对照，按连续区间划分查询
void search_std_thread(const IVFIndex& index, const float* queries, std::size_t count,
                       const SearchOptions& options, Neighbor* output) {
    if (count == 0) return;

    // 根据配置选择距离计算函数
    auto distance = options.use_simd ? l2_distance_simd : l2_distance_scalar;

    // 实际创建的线程数不超过查询数
    const std::size_t thread_count = std::min<std::size_t>(count, std::size_t(std::max(options.threads, 1)));

    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    // 每个线程处理一个连续查询区间
    const std::size_t base = count / thread_count;
    const std::size_t extra = count % thread_count;

    for (std::size_t t = 0; t < thread_count; ++t) {
        const std::size_t begin = t * base + std::min<std::size_t>(t, extra);
        const std::size_t len = base + (t < extra ? 1 : 0);
        const std::size_t end = begin + len;

        workers.emplace_back([&, begin, end]() {
            for (std::size_t q = begin; q < end; ++q) {
                const float* query = queries + q * index.dim;
                const auto lists = select_lists(index, query, options.nprobe);
                TopK best(options.k);

                // 扫描选中的倒排列表，维护当前查询的Top-k
                for (auto list : lists) {
                    for (std::size_t p = index.list_ptr[list];
                         p < index.list_ptr[list + 1]; ++p) {
                        best.add({distance(query, index.vector(p), index.dim), index.ids[p]});
                    }
                }

                best.finish(output + q * options.k);
            }
        });
    }

    // 等待所有线程结束
    for (auto& worker : workers) {
        worker.join();
    }
}

// 查询级并行，支持static/dynamic/guided
void search_omp_student_1(const IVFIndex& index, const float* queries, std::size_t count,
                          const SearchOptions& options, Neighbor* output) {
    #pragma omp parallel for num_threads(options.threads) schedule(runtime)
    for (std::size_t q = 0; q < count; q++) {
        // 取第q个查询向量
        const float* query = queries + q * index.dim;
        // 根据配置选择距离计算函数
        const auto distance = options.use_simd ? l2_distance_simd : l2_distance_scalar;
        // 找最近的8个聚类列表
        const auto lists = select_lists(index, query, options.nprobe);
        TopK best(options.k);

        // 扫描8个列表中的所有候选，计算欧式距离加入TopK容器
        for (auto list : lists) {
            for (std::size_t p = index.list_ptr[list]; p < index.list_ptr[list + 1]; p++) {
                best.add({distance(query, index.vector(p), index.dim), index.ids[p]});
            }
        }
        // 输出最终Top10结果
        best.finish(output + q * options.k);
    }
}

// 查询内部并行，使用task和局部Top-k
void search_omp_student_2(const IVFIndex& index, const float* queries, std::size_t count,
                          const SearchOptions& options, Neighbor* output) {
    // 创建一个OpenMP线程团队，在团队内部生成task
    #pragma omp parallel num_threads(options.threads)
    {
        #pragma omp single
        {
            for (std::size_t q = 0; q < count; q++) {
                const float* query = queries + q * index.dim;
                // 根据配置选择距离计算函数
                const auto distance = options.use_simd ? l2_distance_simd : l2_distance_scalar;
                const auto lists = select_lists(index, query, options.nprobe);

                // 按grain划分任务，每个task负责一段候选向量
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

                // 每个task维护独立的局部Top-k
                std::vector<std::vector<Neighbor>> local_results(chunks.size());

                #pragma omp taskgroup
                {
                    for (std::size_t i = 0; i < chunks.size(); i++) {
                        #pragma omp task firstprivate(i, query, distance) shared(chunks, local_results, index, options)
                        {
                            TopK local_best(options.k);
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

                // 所有task结束后，合并局部Top-k
                TopK best(options.k);
                for (const auto& local : local_results) {
                    for (const auto& neighbor : local) {
                        if (neighbor.id != -1) {
                            best.add(neighbor);
                        }
                    }
                }

                best.finish(output + q * options.k);
            }
        }
    }
}
}