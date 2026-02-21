#include <Eigen/Core>
#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <numeric>
#include <random>
#include <spdlog/spdlog.h>
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

struct ArgParseResult {
  int exit_code{0};
  bool should_exit{false};
};

std::vector<int> parse_threads(const std::string &value) {
  std::vector<int> out;
  const std::string &s = value;
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

ArgParseResult parse_args(int argc, char **argv, BenchConfig &cfg) {
  std::string thread_counts_csv = "1,2,4,8";
  CLI::App app{"vctk_bench_vdp: benchmark VDP on synthetic data"};
  app.add_option("--n", cfg.n, "number of samples")->check(CLI::PositiveNumber);
  app.add_option("--d", cfg.d, "feature dimension")->check(CLI::PositiveNumber);
  app.add_option("--k", cfg.k_true, "true synthetic clusters")
      ->check(CLI::PositiveNumber);
  app.add_option("--reps", cfg.reps, "repeated runs per thread count")
      ->check(CLI::PositiveNumber);
  app.add_option("--max-clusters", cfg.max_clusters, "max VDP clusters")
      ->check(CLI::PositiveNumber);
  app.add_option("--max-iters", cfg.max_iters, "max VBEM iterations")
      ->check(CLI::PositiveNumber);
  app.add_option("--split-refine-iters", cfg.split_refine_iters,
                 "split refine iterations")
      ->check(CLI::PositiveNumber);
  app.add_option("--threads", thread_counts_csv,
                 "thread counts CSV, e.g. 1,2,4,8");
  app.add_option("--csv", cfg.csv_path, "output CSV path");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return ArgParseResult{app.exit(e), true};
  }

  try {
    cfg.thread_counts = parse_threads(thread_counts_csv);
  } catch (const std::exception &e) {
    spdlog::error("invalid --threads value '{}': {}", thread_counts_csv,
                  e.what());
    return ArgParseResult{1, true};
  }

  return ArgParseResult{};
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
    throw std::runtime_error(
        fmt::format("failed to open benchmark CSV for writing: {}", csv_path));
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
  spdlog::info("Summary (median elapsed ms)");
  for (int t : thread_counts) {
    std::vector<double> times;
    for (const auto &r : rows) {
      if (r.threads == t) {
        times.push_back(r.elapsed_ms);
      }
    }
    if (!times.empty()) {
      spdlog::info("  threads={} median_ms={}", t, median_ms(times));
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  BenchConfig cfg;
  try {
    const ArgParseResult parse_result = parse_args(argc, argv, cfg);
    if (parse_result.should_exit) {
      return parse_result.exit_code;
    }
    const Eigen::MatrixXd X = make_synthetic(cfg);

    spdlog::info("Benchmarking VDP on synthetic data");
    spdlog::info("N={} D={} K_true={} reps={}", cfg.n, cfg.d, cfg.k_true,
                 cfg.reps);
    std::string thread_list;
    for (const int t : cfg.thread_counts) {
      if (!thread_list.empty()) {
        thread_list += ",";
      }
      thread_list += std::to_string(t);
    }
    spdlog::info("Threads: {}", thread_list);

    std::vector<BenchResult> rows;
    rows.reserve(static_cast<std::size_t>(cfg.reps * cfg.thread_counts.size()));

    for (int t : cfg.thread_counts) {
      for (int r = 0; r < cfg.reps; ++r) {
        const BenchResult row = run_once(X, cfg, t, r);
        rows.push_back(row);

        spdlog::info("  run threads={} rep={}/{} elapsed_ms={} clusters={}", t,
                     (r + 1), cfg.reps, row.elapsed_ms, row.clusters_found);
      }
    }

    write_csv(cfg.csv_path, rows);
    print_summary(rows, cfg.thread_counts);

    spdlog::info("Wrote benchmark CSV: {}", cfg.csv_path);
    return 0;
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    return 1;
  }
}
