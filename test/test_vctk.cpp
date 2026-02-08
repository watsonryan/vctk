#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "distributions.hpp"
#include "vctk.hpp"

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;

void assert_prob_rows(const MatrixXd &qZ) {
  for (Eigen::Index i = 0; i < qZ.rows(); ++i) {
    const double s = qZ.row(i).sum();
    assert(std::isfinite(s));
    assert(std::abs(s - 1.0) < 1e-6);
  }
}

MatrixXd make_gauss_data() {
  MatrixXd X(8, 2);
  X << -2.0, -2.1, -1.9, -2.0, -2.2, -1.8, -1.7, -2.3, // cluster A
      2.0, 2.1, 1.9, 2.2, 2.3, 1.8, 2.1, 1.7;           // cluster B
  return X;
}

MatrixXd make_nonneg_data() {
  MatrixXd X(8, 2);
  X << 0.2, 0.3, 0.4, 0.3, 0.25, 0.35, 0.45, 0.4, 2.0, 2.2, 2.1, 2.3, 2.4,
      1.9, 2.2, 2.0;
  return X;
}

template <class W>
void assert_common(const MatrixXd &qZ, const W &weights, std::size_t n_clusters,
                   double f) {
  assert(std::isfinite(f));
  assert(qZ.rows() > 0);
  assert(qZ.cols() > 0);
  assert_prob_rows(qZ);
  assert(static_cast<std::size_t>(qZ.cols()) == n_clusters);
  assert(static_cast<std::size_t>(weights.Elogweight().size()) == n_clusters);
}

void test_vdp() {
  const MatrixXd X = make_gauss_data();

  distributions::StickBreak weights;
  std::vector<distributions::GaussWish> clusters;
  MatrixXd qZ;

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 50;
  opts.split_refine_iters = 10;

  const double f = vctk::learnVDP(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
  assert_common(qZ, weights, clusters.size(), f);
}

void test_vdp_threads_option() {
  const MatrixXd X = make_gauss_data();

  distributions::StickBreak weights;
  std::vector<distributions::GaussWish> clusters;
  MatrixXd qZ;

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 50;
  opts.split_refine_iters = 10;
  opts.n_threads = 2;

  const double f = vctk::learnVDP(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
  assert_common(qZ, weights, clusters.size(), f);
}

void test_bgmm() {
  const MatrixXd X = make_gauss_data();

  distributions::Dirichlet weights;
  std::vector<distributions::GaussWish> clusters;
  MatrixXd qZ;

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 50;
  opts.split_refine_iters = 10;

  const double f =
      vctk::learnBGMM(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
  assert_common(qZ, weights, clusters.size(), f);
}

void test_dgmm() {
  const MatrixXd X = make_gauss_data();

  distributions::Dirichlet weights;
  std::vector<distributions::NormGamma> clusters;
  MatrixXd qZ;

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 50;
  opts.split_refine_iters = 10;

  const double f =
      vctk::learnDGMM(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
  assert_common(qZ, weights, clusters.size(), f);
}

void test_bemm() {
  const MatrixXd X = make_nonneg_data();

  distributions::Dirichlet weights;
  std::vector<distributions::ExpGamma> clusters;
  MatrixXd qZ;

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 50;
  opts.split_refine_iters = 10;

  const double f =
      vctk::learnBEMM(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
  assert_common(qZ, weights, clusters.size(), f);
}

void test_bemm_rejects_negative_data() {
  MatrixXd X(2, 1);
  X << -1.0, 0.1;

  distributions::Dirichlet weights;
  std::vector<distributions::ExpGamma> clusters;
  MatrixXd qZ;

  bool threw = false;
  try {
    (void)vctk::learnBEMM(X, qZ, weights, clusters);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

void test_vdp_rejects_nonfinite_data() {
  MatrixXd X(4, 2);
  X << 0.0, 1.0, 2.0, 3.0, 4.0, std::numeric_limits<double>::infinity(), 6.0,
      7.0;

  distributions::StickBreak weights;
  std::vector<distributions::GaussWish> clusters;
  MatrixXd qZ;

  bool threw = false;
  try {
    (void)vctk::learnVDP(X, qZ, weights, clusters);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

void test_vdp_deterministic_repeatable() {
  const MatrixXd X = make_gauss_data();

  distributions::StickBreak w1;
  distributions::StickBreak w2;
  std::vector<distributions::GaussWish> c1;
  std::vector<distributions::GaussWish> c2;
  MatrixXd q1;
  MatrixXd q2;

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 60;
  opts.split_refine_iters = 12;
  opts.deterministic = true;
  opts.n_threads = 4; // ignored under deterministic mode

  const double f1 = vctk::learnVDP(X, q1, w1, c1, vctk::PRIORVAL, opts);
  const double f2 = vctk::learnVDP(X, q2, w2, c2, vctk::PRIORVAL, opts);

  assert(std::abs(f1 - f2) < 1.0e-9);
  assert(q1.rows() == q2.rows());
  assert(q1.cols() == q2.cols());
  assert((q1 - q2).norm() < 1.0e-9);
  assert(c1.size() == c2.size());
}

void test_progress_callback_invoked() {
  const MatrixXd X = make_gauss_data();
  distributions::StickBreak weights;
  std::vector<distributions::GaussWish> clusters;
  MatrixXd qZ;

  int calls = 0;
  int last_clusters = 0;
  double last_f = std::numeric_limits<double>::quiet_NaN();

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 40;
  opts.split_refine_iters = 8;
  opts.progress_callback = [&](const vctk::LearnProgress &p) {
    ++calls;
    last_clusters = p.clusters;
    last_f = p.free_energy;
  };

  (void)vctk::learnVDP(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
  assert(calls > 0);
  assert(last_clusters > 0);
  assert(std::isfinite(last_f));
}

void test_vdp_golden_cluster_count() {
  MatrixXd X(80, 2);
  for (int i = 0; i < 40; ++i) {
    X(i, 0) = -5.0 + 0.03 * i;
    X(i, 1) = -5.0 - 0.02 * i;
  }
  for (int i = 0; i < 40; ++i) {
    X(40 + i, 0) = 5.0 + 0.02 * i;
    X(40 + i, 1) = 5.0 - 0.03 * i;
  }
  distributions::StickBreak w1;
  distributions::StickBreak w2;
  std::vector<distributions::GaussWish> c1;
  std::vector<distributions::GaussWish> c2;
  MatrixXd q1;
  MatrixXd q2;

  vctk::VbemOptions opts;
  opts.max_clusters = 4;
  opts.max_vbem_iters = 80;
  opts.split_refine_iters = 10;
  opts.deterministic = true;

  const double f1 = vctk::learnVDP(X, q1, w1, c1, vctk::PRIORVAL, opts);
  const double f2 = vctk::learnVDP(X, q2, w2, c2, vctk::PRIORVAL, opts);
  assert(std::isfinite(f1));
  assert(std::isfinite(f2));
  assert(c1.size() == c2.size());
  assert(std::abs(f1 - f2) < 1.0e-9);
  assert((q1 - q2).norm() < 1.0e-9);
}

void test_parallel_independent_runs() {
  auto runner = []() {
    const MatrixXd X = make_gauss_data();
    distributions::StickBreak weights;
    std::vector<distributions::GaussWish> clusters;
    MatrixXd qZ;
    vctk::VbemOptions opts;
    opts.max_clusters = 4;
    opts.max_vbem_iters = 50;
    opts.split_refine_iters = 10;
    opts.n_threads = 2;
    const double f = vctk::learnVDP(X, qZ, weights, clusters, vctk::PRIORVAL, opts);
    assert(std::isfinite(f));
    assert(!clusters.empty());
  };

  std::thread t1(runner);
  std::thread t2(runner);
  std::thread t3(runner);
  t1.join();
  t2.join();
  t3.join();
}

void test_strong_exception_guarantee_on_callback_throw() {
  const MatrixXd X = make_gauss_data();

  MatrixXd q_before;
  distributions::StickBreak w_before;
  std::vector<distributions::GaussWish> c_before;
  vctk::VbemOptions ok_opts;
  ok_opts.max_clusters = 4;
  ok_opts.max_vbem_iters = 50;
  ok_opts.split_refine_iters = 10;
  (void)vctk::learnVDP(X, q_before, w_before, c_before, vctk::PRIORVAL, ok_opts);

  MatrixXd q = q_before;
  distributions::StickBreak w = w_before;
  std::vector<distributions::GaussWish> c = c_before;

  int error_cb_calls = 0;
  vctk::VbemOptions bad_opts = ok_opts;
  bad_opts.progress_callback = [](const vctk::LearnProgress &) {
    throw std::runtime_error("intentional callback failure");
  };
  bad_opts.error_callback = [&](const vctk::ErrorContext &ctx) {
    ++error_cb_calls;
    assert(ctx.algorithm == "learnVDP");
    assert(ctx.phase == "progress_callback");
  };

  bool threw = false;
  try {
    (void)vctk::learnVDP(X, q, w, c, vctk::PRIORVAL, bad_opts);
  } catch (const vctk::RuntimeError &e) {
    assert(e.code() == vctk::ErrorCode::CallbackFailure);
    threw = true;
  }
  assert(threw);
  assert(error_cb_calls > 0);

  assert(q.rows() == q_before.rows());
  assert(q.cols() == q_before.cols());
  assert((q - q_before).norm() < 1.0e-12);

  const Eigen::ArrayXd ew = w.Elogweight();
  const Eigen::ArrayXd ew_before = w_before.Elogweight();
  assert(ew.size() == ew_before.size());
  assert((ew - ew_before).matrix().norm() < 1.0e-12);
  assert(c.size() == c_before.size());
}

} // namespace

int main() {
  test_vdp();
  test_vdp_threads_option();
  test_bgmm();
  test_dgmm();
  test_bemm();
  test_bemm_rejects_negative_data();
  test_vdp_rejects_nonfinite_data();
  test_vdp_deterministic_repeatable();
  test_progress_callback_invoked();
  test_vdp_golden_cluster_count();
  test_parallel_independent_runs();
  test_strong_exception_guarantee_on_callback_throw();

  std::cout << "All vctk algorithm tests passed\n";
  return 0;
}
