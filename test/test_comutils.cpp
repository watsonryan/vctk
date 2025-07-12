#include <cassert>
#include <iostream>

#include <Eigen/Dense>

#include "comutils.hpp"      // must come *before* the using‑directive

namespace {

using comutils::ArrayXb;
using Eigen::ArrayXi;
using Eigen::Index;
using Eigen::MatrixXd;

constexpr double eps = 1e-12;

// ------------------------------------------------------------------
// 1. arrfind
// ------------------------------------------------------------------
void test_arrfind() {
  ArrayXb mask(5);
  mask << true, false, true, false, false;

  ArrayXi idx_true, idx_false;
  comutils::arrfind(mask, idx_true, idx_false);

  assert(idx_true.size() == 2 && idx_true(0) == 0 && idx_true(1) == 2);
  assert(idx_false.size() == 3 && idx_false(0) == 1 && idx_false(1) == 3 &&
         idx_false(2) == 4);
}

// ------------------------------------------------------------------
// 2. partobs
// ------------------------------------------------------------------
void test_partobs() {
  MatrixXd X(5, 1);
  X << 10, 20, 30, 40, 50;

  ArrayXb mask(5);
  mask << true, false, true, false, false;

  MatrixXd Xk;
  ArrayXi pidx = comutils::partobs(X, mask, Xk);

  assert(pidx.size() == 2 && pidx(0) == 0 && pidx(1) == 2);
  assert(Xk.rows() == 2 && Xk(0, 0) == 10 && Xk(1, 0) == 30);
}

// ------------------------------------------------------------------
// 3. auglabels
// ------------------------------------------------------------------
void test_auglabels() {
  constexpr Index N = 4; // observations
  constexpr Index K = 2; // original clusters

  ArrayXi map(N);
  for (Index i = 0; i < N; ++i)
    map(i) = i;

  // Build a bool mask with first two observations selected
  ArrayXb Zsplit = ArrayXb::Constant(N, false);
  Zsplit.head(2).setConstant(true);

  MatrixXd qZ(N, K);
  qZ << 0.80, 0.20, 0.75, 0.25, 0.60, 0.40, 0.30, 0.70;

  const MatrixXd qZaug = comutils::auglabels(/*k=*/0, map, Zsplit, qZ);

  assert(qZaug.rows() == N && qZaug.cols() == K + 1);

  assert(std::abs(qZaug(0, 0) - 0.0) < eps);
  assert(std::abs(qZaug(0, 2) - 0.80) < eps);
  assert(std::abs(qZaug(1, 0) - 0.0) < eps);
  assert(std::abs(qZaug(1, 2) - 0.75) < eps);

  assert(std::abs(qZaug(2, 0) - 0.60) < eps);
  assert(std::abs(qZaug(3, 0) - 0.30) < eps);
}

} // namespace

int main() {
  test_arrfind();
  test_partobs();
  test_auglabels();

  std::cout << "All comutils tests passed ✔️\n";
  return 0;
}
