#include "ivf_search.h"

#include <omp.h>
#include <cmath>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

struct Arguments {
    std::string base = "/var/dataset/sift_base.fbin";
    std::string centroids = "/var/dataset/centroid_4096.fvecs";
    std::string cluster_ids = "/var/dataset/cluster_id_4096.ivecs";
    std::string queries_file = "/var/dataset/sift_query.fbin";
    std::string groundtruth = "/var/dataset/gt.ibin";
    std::string only = "serial", mode = "bench", csv;
    std::string query_order = "original";
    int perf_control_fd = -1, perf_ack_fd = -1;
    std::size_t queries = 128, batch = 32, repeats = 10, warmup = 1;
    ivf::SearchOptions search;
};

std::size_t number(const std::string& value) {
    std::size_t n;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), n);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid integer: " + value);
    }
    return n;
}

Arguments parse(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[i + 1];
        if (key == "--base") args.base = value;
        else if (key == "--centroids") args.centroids = value;
        else if (key == "--cluster-ids") args.cluster_ids = value;
        else if (key == "--queries-file") args.queries_file = value;
        else if (key == "--gt") args.groundtruth = value;
        else if (key == "--only") args.only = value;
        else if (key == "--mode") args.mode = value;
        else if (key == "--csv") args.csv = value;
        else if (key == "--query-order") args.query_order = value;
        else if (key == "--perf-control-fd" || key == "--perf-ack-fd") {
            const auto fd = number(value);
            if (fd > INT_MAX) throw std::invalid_argument("invalid perf FD");
            (key == "--perf-control-fd" ? args.perf_control_fd : args.perf_ack_fd) = static_cast<int>(fd);
        }
        else if (key == "--distance") {
            if (value == "scalar") args.search.use_simd = false;
            else if (value == "simd") args.search.use_simd = true;
            else throw std::invalid_argument("--distance expects scalar or simd");
        }
        else if (key == "--queries") args.queries = number(value);
        else if (key == "--batch") args.batch = number(value);
        else if (key == "--repeats") args.repeats = number(value);
        else if (key == "--warmup") args.warmup = number(value);
        else if (key == "--k") args.search.k = number(value);
        else if (key == "--nprobe") args.search.nprobe = number(value);
        else if (key == "--grain") args.search.grain = number(value);
        else if (key == "--threads") {
            const auto n = number(value);
            if (n == 0 || n > INT_MAX) throw std::invalid_argument("invalid thread count");
            args.search.threads = static_cast<int>(n);
        } else throw std::invalid_argument("unknown option: " + key);
    }
    if (!args.queries || !args.batch || !args.repeats || !args.search.k ||
        !args.search.nprobe || !args.search.grain) {
        throw std::invalid_argument("counts must be positive (except warmup)");
    }
    if (args.mode != "check" && args.mode != "bench") {
        throw std::invalid_argument("--mode expects check or bench");
    }
    if (args.query_order != "original" && args.query_order != "sorted") {
        throw std::invalid_argument("--query-order expects original or sorted");
    }
    if ((args.perf_control_fd >= 0) != (args.perf_ack_fd >= 0) ||
        (args.perf_control_fd >= 0 &&
         (args.perf_control_fd == args.perf_ack_fd || args.mode != "bench"))) {
        throw std::invalid_argument("perf requires two distinct FDs and --mode bench");
    }
    return args;
}

// 发送完整控制命令，并在有限时间内等待perf确认
void perf_control(int control_fd, int ack_fd, const std::string& command) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto wait_ready = [&](int fd, short events) {
        for (;;) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            if (remaining <= 0) throw std::runtime_error("perf control timeout: " + command);
            pollfd item{fd, events, 0};
            const int result = poll(&item, 1, static_cast<int>(remaining));
            if (result < 0 && errno == EINTR) continue;
            if (result < 0) throw std::runtime_error("perf control poll failed");
            if (result == 0) continue;
            if (item.revents & events) return;
            throw std::runtime_error("perf control channel disconnected");
        }
    };
    const std::string message = command + '\n';
    std::size_t sent = 0;
    while (sent < message.size()) {
        wait_ready(control_fd, POLLOUT);
        const auto n = write(control_fd, message.data() + sent, message.size() - sent);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (n <= 0) throw std::runtime_error("perf control write failed");
        sent += static_cast<std::size_t>(n);
    }
    std::string reply;
    for (;;) {
        wait_ready(ack_fd, POLLIN);
        char c;
        const auto n = read(ack_fd, &c, 1);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (n <= 0) throw std::runtime_error("perf ack channel closed");
        // perf可能在ack换行后附带NUL终止符
        if (c == '\n' || c == '\0' || c == '\r') {
            if (reply.empty()) continue;
            if (reply == "ack") return;
            throw std::runtime_error("unexpected perf ack");
        }
        reply += c;
        if (reply.size() > 3 || std::string("ack").compare(0, reply.size(), reply) != 0) {
            throw std::runtime_error("unexpected perf ack");
        }
    }
}

ivf::SearchFunction select_strategy(const std::string& name) {
    if (name == "serial") return ivf::search_ivf_serial;
    if (name == "std") return ivf::search_std_thread;
    if (name == "exact") return ivf::search_exact_serial;
    if (name == "omp1") return ivf::search_omp_student_1;
    if (name == "omp2") return ivf::search_omp_student_2;
    throw std::invalid_argument("unknown strategy: " + name);
}

int run(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "SIFT IVF-Flat starter\n"
            "  --only serial|std|exact|omp1|omp2   --mode check|bench\n"
            "  --queries 128   --batch 32   --k 10   --nprobe 8\n"
            "  --threads 4   --grain 1024   --repeats 10   --warmup 1\n"
            "  --base PATH (.fbin)   --centroids PATH (.fvecs)\n"
            "  --cluster-ids PATH (.ivecs)   --queries-file PATH (.fbin)\n"
            "  --query-order original|sorted\n"
            "  --distance scalar|simd\n"
            "  --perf-control-fd FD   --perf-ack-fd FD\n"
            "  --gt PATH (.ibin)   --csv PATH\n";
        return 0;
    }
    auto args = parse(argc, argv);
    if (args.perf_control_fd >= 0) {
        // 非阻塞FD配合poll超时，避免通信中断后永久等待
        for (int fd : {args.perf_control_fd, args.perf_ack_fd}) {
            const int flags = fcntl(fd, F_GETFL);
            if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
                throw std::invalid_argument("invalid perf control FD");
            }
        }
        if (std::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
            throw std::runtime_error("cannot handle perf pipe disconnect");
        }
    }
    const auto search = select_strategy(args.only);
    const auto index = ivf::read_ivf(args.base, args.centroids, args.cluster_ids);
    auto queries = ivf::read_fbin(args.queries_file);
    auto truth = ivf::read_ibin(args.groundtruth);
    if (args.only == "exact") args.search.nprobe = index.nlist;
    if (args.only == "serial" || args.only == "exact") args.search.threads = 1;
    if (queries.dim != index.dim || args.queries > queries.rows || args.queries > truth.rows ||
        args.search.k > index.size || args.search.k > truth.dim || args.search.nprobe > index.nlist) {
        throw std::invalid_argument("query count, dimension, k or nprobe exceeds input bounds");
    }
    for (std::size_t q = 0; q < args.queries; ++q) {
        const auto* row = truth.row(q);
        for (std::size_t j = 0; j < args.search.k; ++j) {
            if (row[j] < 0 || std::size_t(row[j]) >= index.size ||
                std::find(row, row + j, row[j]) != row + j) {
                throw std::invalid_argument("invalid or duplicate ground truth ID");
            }
        }
    }
    if (args.query_order == "sorted") {
        // 计算每个查询需要扫描的候选数量C(q)
        std::vector<std::size_t> candidate_counts(args.queries);
        for (std::size_t q = 0; q < args.queries; ++q) {
            const auto lists = ivf::select_lists(index, queries.row(q), args.search.nprobe);
            for (const auto list : lists) {
                candidate_counts[q] += index.list_ptr[list + 1] - index.list_ptr[list];
            }
        }

        // 按候选数升序排列查询编号
        std::vector<std::size_t> order(args.queries);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            if (candidate_counts[a] != candidate_counts[b]) {
                return candidate_counts[a] < candidate_counts[b];
            }
            return a < b;
        });

        // 同步重排查询向量和ground truth
        ivf::FloatMatrix sorted_queries;
        sorted_queries.rows = args.queries;
        sorted_queries.dim = queries.dim;
        sorted_queries.data.resize(args.queries * queries.dim);
        ivf::IntMatrix sorted_truth;
        sorted_truth.rows = args.queries;
        sorted_truth.dim = truth.dim;
        sorted_truth.data.resize(args.queries * truth.dim);
        for (std::size_t q = 0; q < args.queries; ++q) {
            std::copy_n(queries.row(order[q]), queries.dim,
                        sorted_queries.data.data() + q * queries.dim);
            std::copy_n(truth.row(order[q]), truth.dim,
                        sorted_truth.data.data() + q * truth.dim);
        }
        queries = std::move(sorted_queries);
        truth = std::move(sorted_truth);
    }
    omp_set_dynamic(0);
    omp_set_num_threads(args.search.threads);
    std::cout << "N=" << index.size << " dim=" << index.dim << " nlist=" << index.nlist
              << " strategy=" << args.only << " threads=" << args.search.threads << '\n';

    std::vector<ivf::Neighbor> reference(args.queries * args.search.k), output(reference.size());
    const auto reference_search = args.search.nprobe == index.nlist
        ? ivf::search_exact_serial : ivf::search_ivf_serial;
    reference_search(index, queries.data.data(), args.queries, args.search, reference.data());
    const double recall = ivf::recall_at_k(reference, truth, args.queries, args.search.k);
    
    std::vector<double> latencies;
    std::vector<double> batch_times;
    bool collect_perf = false;
    
    auto run_once = [&] {
        std::fill(output.begin(), output.end(),
                  ivf::Neighbor{std::numeric_limits<double>::quiet_NaN(), -2});
        double ms = 0;
        batch_times.clear();
        for (std::size_t begin = 0; begin < args.queries; begin += args.batch) {
            const auto count = std::min(args.batch, args.queries - begin);
            if (collect_perf) perf_control(args.perf_control_fd, args.perf_ack_fd, "enable");
            const auto start = std::chrono::steady_clock::now();
            search(index, queries.row(begin), count, args.search, output.data() + begin * args.search.k);
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (collect_perf) perf_control(args.perf_control_fd, args.perf_ack_fd, "disable");
            ms += elapsed;
            batch_times.push_back(elapsed);

            // 仅在batch=1时记录单查询延迟
            if (args.batch == 1) {
                latencies.push_back(elapsed);
            }
        }
        ivf::check_result(output, reference);
        return ms;
    };
    std::cout << std::fixed << std::setprecision(6);
    if (args.mode == "check") {
        run_once();
        std::cout << "correctness=PASS recall@" << args.search.k << '=' << recall << '\n';
        return 0;
    }

    std::ofstream csv;
    if (!args.csv.empty()) {
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv.open(args.csv);
        csv << "strategy,threads,queries,batch,k,nprobe,grain,repeat,time_ms,qps,recall,correctness,p50_ms,p95_ms,batch_times_ms,query_order\n"
            << std::setprecision(12);
    }
    for (std::size_t r = 0; r < args.warmup; ++r) run_once();
    // 仅对正式batch检索开启计数，确认等待位于原计时区间之外
    collect_perf = args.perf_control_fd >= 0;
    latencies.clear(); // 不统计预热阶段的延迟
    std::vector<double> times(args.repeats);
    for (std::size_t r = 0; r < args.repeats; ++r) {
        times[r] = run_once();
        double p50 = 0, p95 = 0;
        if (args.batch == 1) {
            std::sort(latencies.begin(), latencies.end());

            auto percentile = [&](double p) {
                const std::size_t i = static_cast<std::size_t>(
                    std::ceil(p * latencies.size())) - 1;
                return latencies[i];
            };

            p50 = percentile(0.50);
            p95 = percentile(0.95);
            latencies.clear();
        }
        if (csv.is_open()) {
            csv << args.only << ',' << args.search.threads << ',' << args.queries << ','
                << args.batch << ',' << args.search.k << ',' << args.search.nprobe << ','
                << args.search.grain << ',' << r + 1 << ',' << times[r] << ','
                << args.queries * 1000.0 / times[r] << ',' << recall << ",PASS,";
            if (args.batch == 1) {
                csv << p50 << ',' << p95;
            } else {
                csv << ',';
            }

            csv << ",\"";
            for (std::size_t i = 0; i < batch_times.size(); ++i) {
                if (i != 0) csv << ';';
                csv << batch_times[i];
            }
            csv << "\"," << args.query_order << '\n';
        }
    }
    if (csv.is_open()) csv.close();
    std::sort(times.begin(), times.end());
    const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
    std::cout << "correctness=PASS recall@" << args.search.k << '=' << recall << '\n'
              << "time_ms[min/median/max]=" << times.front() << '/' << median << '/' << times.back()
              << " qps=" << args.queries * 1000.0 / median << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
