#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Dense>

#include "probutils.hpp"

namespace {

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using probutils::Index;

constexpr double eps = 1e-10; // tolerance for floating‑point comparisons

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------
bool eq(double a, double b, double tol = eps) { return std::abs(a - b) < tol; }

// -----------------------------------------------------------------------------
// 1. mean / stdev / cov
// -----------------------------------------------------------------------------
void test_basic_stats() {
  MatrixXd X(4, 2);
  X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;

  RowVectorXd m = probutils::mean(X);
  assert(eq(m(0), 4.0) && eq(m(1), 5.0));

  RowVectorXd sd = probutils::stdev(X);
  assert(eq(sd(0), std::sqrt(20.0 / 3.0)));
  assert(eq(sd(1), std::sqrt(20.0 / 3.0)));

  MatrixXd Cv = probutils::cov(X);
  assert(eq(Cv(0, 0), 20.0 / 3.0));
  assert(eq(Cv(1, 1), 20.0 / 3.0));
}

// -----------------------------------------------------------------------------
// 2. mean / cov over vector<Matrix>
// -----------------------------------------------------------------------------
void test_group_stats() {
  MatrixXd A(2, 2);
  A << 1, 2, 3, 4;
  MatrixXd B(1, 2);
  B << 5, 6;
  std::vector<MatrixXd> vec{A, B};

  RowVectorXd m = probutils::mean(vec);
  assert(eq(m(0), 3.0) && eq(m(1), 4.0));

  MatrixXd C = probutils::cov(vec);
  assert(C.rows() == 2 && C.cols() == 2);
}

// -----------------------------------------------------------------------------
// 3. logsumexp
// -----------------------------------------------------------------------------
void test_logsumexp() {
  MatrixXd X(2, 3);
  X << std::log(1.0), std::log(2.0), std::log(3.0), 10.0, 11.0, 12.0;

  VectorXd lse = probutils::logsumexp(X);
  assert(eq(lse(0), std::log(1 + 2 + 3)));
  assert(eq(lse(1), 12.0 + std::log(1 + std::exp(-1) + std::exp(-2))));
}

// -----------------------------------------------------------------------------
// 4. Mahalanobis distance
// -----------------------------------------------------------------------------
void test_mahalanobis() {
  MatrixXd X(2, 2);
  X << 1, 0, 0, 1;
  RowVectorXd mu(2);
  mu << 0, 0;
  MatrixXd A = MatrixXd::Identity(2, 2);

  VectorXd d = probutils::mahaldist(X, mu, A);
  assert(eq(d(0), 1.0));
  assert(eq(d(1), 1.0));
}

// -----------------------------------------------------------------------------
// 5. eigpower / logdet
// -----------------------------------------------------------------------------
void test_eig_helpers() {
  MatrixXd M(2, 2);
  M << 2, 0, 0, 3;

  Eigen::VectorXd eigvec;
  double eigval = probutils::eigpower(M, eigvec);
  assert(eq(eigval, 3.0));
  assert(eq(eigvec.norm(), 1.0));

  double ld = probutils::logdet(M);
  assert(eq(ld, std::log(6.0)));

  MatrixXd Z = MatrixXd::Zero(2, 2);
  double zero_eval = probutils::eigpower(Z, eigvec);
  assert(eq(zero_eval, 0.0));
  assert(eq(eigvec.norm(), 1.0));
}

// -----------------------------------------------------------------------------
// 6. digamma / lgamma helpers
// -----------------------------------------------------------------------------
void test_special_functions() {
  MatrixXd X(2, 2);
  X << 1.0, 2.0, 3.5, 5.0;

  MatrixXd lg = probutils::mxlgamma(X);
  for (Index i = 0; i < X.size(); ++i)
    assert(eq(lg(i), std::lgamma(X(i))));

  MatrixXd dg = probutils::mxdigamma(X);

  double x = 4.0;
  double psi_x = probutils::mxdigamma(MatrixXd::Constant(1, 1, x))(0, 0);
  double psi_x_plus =
      probutils::mxdigamma(MatrixXd::Constant(1, 1, x + 1.0))(0, 0);
  assert(eq(psi_x_plus, psi_x + 1.0 / x, 1e-8));
}

void test_group_stats_validation() {
  std::vector<MatrixXd> empty_rows{MatrixXd(0, 2), MatrixXd(0, 2)};
  bool threw = false;
  try {
    (void)probutils::mean(empty_rows);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)probutils::cov(empty_rows);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

} // namespace

int main() {
  test_basic_stats();
  test_group_stats();
  test_logsumexp();
  test_mahalanobis();
  test_eig_helpers();
  test_special_functions();
  test_group_stats_validation();

  std::cout << "All probutils tests passed ✔️\n";
  return 0;
}
