#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "distributions.hpp"
#include "vctk.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct BenchConfig {
  int n{10000};
  int d{8};
  int k_true{8};
  int reps{5};
  int max_clusters{24};
  int max_iters{150};
  int split_refine_iters{20};
  std::vector<int> thread_counts{1, 2, 4, 8};
  std::string csv_path{"build/bench/vdp_bench.csv"};
};

struct BenchResult {
  int n{0};
  int d{0};
  int k_true{0};
  int threads{0};
  int run_idx{0};
  int clusters_found{0};
  double free_energy{0.0};
  double elapsed_ms{0.0};
};

int parse_int(const char *value, const std::string &name) {
  if (value == nullptr) {
    throw std::invalid_argument("missing value for " + name);
  }
  const int v = std::stoi(value);
  if (v < 1) {
    throw std::invalid_argument(name + " must be >= 1");
  }
  return v;
}

std::vector<int> parse_threads(const char *value) {
  if (value == nullptr) {
    throw std::invalid_argument("missing value for --threads");
  }

  std::vector<int> out;
  std::string s(value);
  std::size_t start = 0;
  while (start < s.size()) {
    const std::size_t comma = s.find(',', start);
    const std::size_t end = (comma == std::string::npos) ? s.size() : comma;
    if (end == start) {
      throw std::invalid_argument("empty entry in --threads");
    }
    const int t = std::stoi(s.substr(start, end - start));
    if (t < 1) {
      throw std::invalid_argument("thread count must be >= 1");
    }
    out.push_back(t);
    start = end + 1;
  }
  if (out.empty()) {
    throw std::invalid_argument("no thread counts provided");
  }
  return out;
}

BenchConfig parse_args(int argc, char **argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--n") {
      cfg.n = parse_int((i + 1 < argc) ? argv[++i] : nullptr, "--n");
      continue;
    }
    if (arg == "--d") {
      cfg.d = parse_int((i + 1 < argc) ? argv[++i] : nullptr, "--d");
      continue;
    }
    if (arg == "--k") {
      cfg.k_true = parse_int((i + 1 < argc) ? argv[++i] : nullptr, "--k");
      continue;
    }
    if (arg == "--reps") {
      cfg.reps = parse_int((i + 1 < argc) ? argv[++i] : nullptr, "--reps");
      continue;
    }
    if (arg == "--max-clusters") {
      cfg.max_clusters =
          parse_int((i + 1 < argc) ? argv[++i] : nullptr, "--max-clusters");
      continue;
    }
    if (arg == "--max-iters") {
      cfg.max_iters =
          parse_int((i + 1 < argc) ? argv[++i] : nullptr, "--max-iters");
      continue;
    }
    if (arg == "--split-refine-iters") {
      cfg.split_refine_iters =
          parse_int((i + 1 < argc) ? argv[++i] : nullptr, "--split-refine-iters");
      continue;
    }
    if (arg == "--threads") {
      cfg.thread_counts = parse_threads((i + 1 < argc) ? argv[++i] : nullptr);
      continue;
    }
    if (arg == "--csv") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for --csv");
      }
      cfg.csv_path = argv[++i];
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: vctk_bench_vdp [options]\n"
                   "  --n <int>                   number of samples (default 10000)\n"
                   "  --d <int>                   feature dimension (default 8)\n"
                   "  --k <int>                   true synthetic clusters (default 8)\n"
                   "  --reps <int>                repeated runs per thread count (default 5)\n"
                   "  --max-clusters <int>        max VDP clusters (default 24)\n"
                   "  --max-iters <int>           max VBEM iters (default 150)\n"
                   "  --split-refine-iters <int>  split refine iters (default 20)\n"
                   "  --threads <csv>             thread counts, e.g. 1,2,4,8\n"
                   "  --csv <path>                output CSV path\n";
      std::exit(0);
    }
    throw std::invalid_argument("unknown argument: " + arg);
  }
  return cfg;
}

Eigen::MatrixXd make_synthetic(const BenchConfig &cfg) {
  std::mt19937 rng(42);
  std::normal_distribution<double> z(0.0, 1.0);
  std::normal_distribution<double> mu_dist(0.0, 10.0);
  std::uniform_real_distribution<double> scale_dist(0.35, 1.5);

  std::vector<Eigen::VectorXd> means;
  means.reserve(static_cast<std::size_t>(cfg.k_true));
  for (int k = 0; k < cfg.k_true; ++k) {
    Eigen::VectorXd mu(cfg.d);
    for (int j = 0; j < cfg.d; ++j) {
      mu(j) = mu_dist(rng);
    }
    means.push_back(mu);
  }

  std::vector<Eigen::VectorXd> scales;
  scales.reserve(static_cast<std::size_t>(cfg.k_true));
  for (int k = 0; k < cfg.k_true; ++k) {
    Eigen::VectorXd s(cfg.d);
    for (int j = 0; j < cfg.d; ++j) {
      s(j) = scale_dist(rng);
    }
    scales.push_back(s);
  }

  Eigen::MatrixXd X(cfg.n, cfg.d);
  for (int i = 0; i < cfg.n; ++i) {
    const int c = i % cfg.k_true;
    for (int j = 0; j < cfg.d; ++j) {
      X(i, j) = means[static_cast<std::size_t>(c)](j) +
                scales[static_cast<std::size_t>(c)](j) * z(rng);
    }
  }

  return X;
}

BenchResult run_once(const Eigen::MatrixXd &X, const BenchConfig &cfg,
                     const int threads, const int run_idx) {
  vctk::VbemOptions opts;
  opts.max_clusters = cfg.max_clusters;
  opts.max_vbem_iters = cfg.max_iters;
  opts.split_refine_iters = cfg.split_refine_iters;
  opts.n_threads = threads;
  opts.verbose = false;

  Eigen::MatrixXd qZ;
  distributions::StickBreak weights;
  std::vector<distributions::GaussWish> clusters;

  const auto t0 = Clock::now();
  const double free_energy =
      vctk::learnVDP(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
  const auto t1 = Clock::now();

  BenchResult r;
  r.n = static_cast<int>(X.rows());
  r.d = static_cast<int>(X.cols());
  r.k_true = cfg.k_true;
  r.threads = threads;
  r.run_idx = run_idx;
  r.clusters_found = static_cast<int>(clusters.size());
  r.free_energy = free_energy;
  r.elapsed_ms = Ms(t1 - t0).count();
  return r;
}

void write_csv(const std::string &csv_path, const std::vector<BenchResult> &rows) {
  const std::filesystem::path p(csv_path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }

  std::ofstream out(csv_path);
  if (!out) {
    throw std::runtime_error("failed to open benchmark CSV for writing");
  }

  out << "n,d,k_true,threads,run_idx,clusters_found,free_energy,elapsed_ms\n";
  for (const auto &r : rows) {
    out << r.n << ',' << r.d << ',' << r.k_true << ',' << r.threads << ','
        << r.run_idx << ',' << r.clusters_found << ',' << r.free_energy << ','
        << r.elapsed_ms << '\n';
  }
}

double median_ms(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const std::size_t mid = v.size() / 2;
  if (v.size() % 2 == 0) {
    return 0.5 * (v[mid - 1] + v[mid]);
  }
  return v[mid];
}

void print_summary(const std::vector<BenchResult> &rows,
                   const std::vector<int> &thread_counts) {
  std::cout << "\nSummary (median elapsed ms)\n";
  for (int t : thread_counts) {
    std::vector<double> times;
    for (const auto &r : rows) {
      if (r.threads == t) {
        times.push_back(r.elapsed_ms);
      }
    }
    if (!times.empty()) {
      std::cout << "  threads=" << t << " median_ms=" << median_ms(times)
                << '\n';
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const BenchConfig cfg = parse_args(argc, argv);
    const Eigen::MatrixXd X = make_synthetic(cfg);

    std::cout << "Benchmarking VDP on synthetic data\n";
    std::cout << "N=" << cfg.n << " D=" << cfg.d << " K_true=" << cfg.k_true
              << " reps=" << cfg.reps << '\n';
    std::cout << "Threads:";
    for (const int t : cfg.thread_counts) {
      std::cout << ' ' << t;
    }
    std::cout << '\n';

    std::vector<BenchResult> rows;
    rows.reserve(static_cast<std::size_t>(cfg.reps * cfg.thread_counts.size()));

    for (int t : cfg.thread_counts) {
      for (int r = 0; r < cfg.reps; ++r) {
        const BenchResult row = run_once(X, cfg, t, r);
        rows.push_back(row);

        std::cout << "  run threads=" << t << " rep=" << (r + 1) << "/"
                  << cfg.reps << " elapsed_ms=" << row.elapsed_ms
                  << " clusters=" << row.clusters_found << '\n';
      }
    }

    write_csv(cfg.csv_path, rows);
    print_summary(rows, cfg.thread_counts);

    std::cout << "Wrote benchmark CSV: " << cfg.csv_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
