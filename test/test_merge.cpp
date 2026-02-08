#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "merge.hpp"
#include "vctk.hpp"

namespace {

using Eigen::MatrixXd;
using Eigen::Vector2d;

MatrixXd make_data() {
  MatrixXd X(10, 2);
  X << -2.1, -1.9, -2.0, -2.2, -1.8, -2.1, -2.2, -1.7, -1.9, -2.0, 2.0, 1.9,
      2.1, 2.2, 1.8, 2.0, 2.3, 2.2, 1.9, 2.1;
  return X;
}

void test_merge_from_empty_prior() {
  const MatrixXd X = make_data();

  Eigen::MatrixXd qZ;
  distributions::StickBreak w;
  std::vector<distributions::GaussWish> clusters;
  vctk::VbemOptions opts;
  opts.max_clusters = 6;
  opts.max_vbem_iters = 60;
  opts.split_refine_iters = 10;
  (void)vctk::learnVDP(X, qZ, w, clusters, vctk::PRIORVAL, opts);

  const auto gmm =
      vctk::merge::mergeMixtureModel(X, qZ, {}, clusters, w, 0.05, 8);
  assert(!gmm.empty());

  double wsum = 0.0;
  for (const auto &c : gmm) {
    wsum += c.weight;
    assert(c.mean.size() == X.cols());
    assert(c.cov.rows() == X.cols());
    assert(c.cov.cols() == X.cols());
  }
  assert(std::isfinite(wsum));
  assert(std::abs(wsum - 1.0) < 1e-6);
}

void test_merge_and_lookup() {
  vctk::merge::MixtureComponent a;
  a.total_obs = 100;
  a.component_obs = 60;
  a.weight = 0.6;
  a.mean = (Eigen::RowVector2d() << -2.0, -2.0).finished();
  a.cov = MatrixXd::Identity(2, 2) * 0.35;

  vctk::merge::MixtureComponent b;
  b.total_obs = 100;
  b.component_obs = 40;
  b.weight = 0.4;
  b.mean = (Eigen::RowVector2d() << 2.0, 2.0).finished();
  b.cov = MatrixXd::Identity(2, 2) * 0.4;

  std::vector<vctk::merge::MixtureComponent> gmm = {a, b};
  const auto picked =
      vctk::merge::getMixtureComponent(gmm, Vector2d(-1.9, -2.1));
  assert((picked.component.mean - a.mean).norm() < 1e-9);

  const auto pruned = vctk::merge::pruneMixtureModel(gmm, 1);
  assert(pruned.size() == 1);
  assert(std::abs(pruned[0].weight - 1.0) < 1e-9);
}

} // namespace

int main() {
  test_merge_from_empty_prior();
  test_merge_and_lookup();
  std::cout << "All merge tests passed\n";
  return 0;
}
