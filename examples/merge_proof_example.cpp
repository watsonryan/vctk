#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <random>
#include <vector>

#include "merge.hpp"
#include "vctk.hpp"

namespace {

Eigen::MatrixXd make_batch(const std::vector<Eigen::Vector2d> &means,
                           const int n_per_cluster, const unsigned int seed) {
  const int k = static_cast<int>(means.size());
  const int n = n_per_cluster * k;

  Eigen::MatrixXd X(n, 2);
  std::mt19937 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);

  const Eigen::Matrix2d chol = (Eigen::Matrix2d() << 0.42, 0.08, 0.00, 0.38).finished();

  int row = 0;
  for (int c = 0; c < k; ++c) {
    for (int i = 0; i < n_per_cluster; ++i) {
      const Eigen::Vector2d u(z(rng), z(rng));
      const Eigen::Vector2d x = means[static_cast<std::size_t>(c)] + chol * u;
      X.row(row++) = x.transpose();
    }
  }

  return X;
}

double min_mean_distance(const vctk::merge::MixtureComponent &a,
                         const std::vector<vctk::merge::MixtureComponent> &pool) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto &b : pool) {
    best = std::min(best, (a.mean - b.mean).norm());
  }
  return best;
}

} // namespace

int main() {
  try {
    const std::vector<Eigen::Vector2d> batch1_means = {
        Eigen::Vector2d(-2.0, -2.0),
        Eigen::Vector2d(2.0, 2.0),
    };
    const std::vector<Eigen::Vector2d> batch2_means = {
        Eigen::Vector2d(-2.02, -1.96), // same semantic cluster as batch1[0]
        Eigen::Vector2d(1.98, 2.05),   // same semantic cluster as batch1[1]
        Eigen::Vector2d(0.2, 4.0),     // new cluster
    };

    const Eigen::MatrixXd X1 = make_batch(batch1_means, 160, 123);
    const Eigen::MatrixXd X2 = make_batch(batch2_means, 140, 456);

    vctk::VbemOptions opts;
    opts.max_clusters = 8;
    opts.max_vbem_iters = 120;
    opts.split_refine_iters = 15;

    Eigen::MatrixXd qZ1;
    Eigen::MatrixXd qZ2;
    distributions::StickBreak w1;
    distributions::StickBreak w2;
    std::vector<distributions::GaussWish> c1;
    std::vector<distributions::GaussWish> c2;

    (void)vctk::learnVDP(X1, qZ1, w1, c1, vctk::PRIORVAL, opts);
    (void)vctk::learnVDP(X2, qZ2, w2, c2, vctk::PRIORVAL, opts);

    const auto prior = vctk::merge::mergeMixtureModel(X1, qZ1, {}, c1, w1, 0.05, 16);
    const auto curr = vctk::merge::mergeMixtureModel(X2, qZ2, {}, c2, w2, 0.05, 16);
    const auto merged = vctk::merge::mergeMixtureModel(X2, qZ2, prior, c2, w2, 0.05, 16);

    int overlap_matches = 0;
    for (const auto &cur : curr) {
      bool matched = false;
      for (const auto &pr : prior) {
        if (vctk::merge::checkComponentEquivalent(pr, cur, 0.05)) {
          matched = true;
          break;
        }
      }
      if (matched) {
        ++overlap_matches;
      }
    }

    spdlog::info("Merge proof run");
    spdlog::info("Batch1 inferred components: {}", prior.size());
    spdlog::info("Batch2 inferred components: {}", curr.size());
    spdlog::info("Equivalent components between batch1 and batch2: {}",
                 overlap_matches);
    spdlog::info("Merged global components: {}", merged.size());
    spdlog::info("Merged component means (x,y):");
    for (std::size_t i = 0; i < merged.size(); ++i) {
      spdlog::info("  [{}] ({}, {}), min_dist_to_batch1={}", i,
                   merged[i].mean(0), merged[i].mean(1),
                   min_mean_distance(merged[i], prior));
    }

    const bool expected = (prior.size() == 2 && curr.size() == 3 && overlap_matches >= 2 &&
                           merged.size() == 3);
    spdlog::info("Expected behavior (2 + 3 with 2 overlaps -> 3 merged): {}",
                 expected ? "PASS" : "CHECK");
    return expected ? 0 : 1;
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    return 1;
  }
}
