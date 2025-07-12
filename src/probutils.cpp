/**
 *  Various statistical helpers
 */

#include "probutils.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

namespace probutils {

namespace {

constexpr double kEigConvThresh = 1.0e-8;
constexpr int kMaxIter = 100;

/* Element‑wise digamma implementation
 * Valid for positive real x.
 */
[[nodiscard]] double digamma_scalar(double x) noexcept {
  // Euler‑Mascheroni γ ≈ 0.57721;
  constexpr double kEulerGamma = 0.57721566490153286060651209;

  if (std::isnan(x) || std::isinf(x))
    return std::numeric_limits<double>::quiet_NaN();

  if (x <= 0.0) {
    // Digamma has poles at non‑positive integers; return NaN
    return std::numeric_limits<double>::quiet_NaN();
  }

  double result = 0.0;
  while (x < 6.0) {
    result -= 1.0 / x;
    x += 1.0;
  }

  // Asymptotic expansion — good to ~1e‑12 for x ≥ 6
  const double inv = 1.0 / x;
  const double inv2 = inv * inv;
  result += std::log(x) - 0.5 * inv -
            inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));

  return result;
}

inline void throw_if(bool cond, const char *msg) {
  if (cond)
    throw std::invalid_argument(msg);
}

[[nodiscard]] inline auto row_mean(Eigen::Ref<const Eigen::MatrixXd> X) {
  return X.colwise().mean();
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  Descriptive statistics
// ─────────────────────────────────────────────────────────────────────────────
Eigen::RowVectorXd mean(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false) {
  throw_if(X.rows() == 0, "mean: empty matrix");
  return row_mean(X);
}

Eigen::RowVectorXd mean(const std::vector<Eigen::MatrixXd> &X) noexcept(false) {
  throw_if(X.empty(), "mean: vector is empty");

  const Index D = X.front().cols();
  Eigen::RowVectorXd sum = Eigen::RowVectorXd::Zero(D);
  Index Ntotal = 0;

  for (const auto &m : X) {
    throw_if(m.cols() != D, "mean: inconsistent dimensions in vector");
    sum += m.colwise().sum();
    Ntotal += m.rows();
  }
  return sum / static_cast<double>(Ntotal);
}

Eigen::RowVectorXd stdev(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false) {
  throw_if(X.rows() < 2, "stdev: need at least two observations");
  const auto mu = row_mean(X);
  return ((X.rowwise() - mu).array().square().colwise().sum() /
          static_cast<double>(X.rows() - 1))
      .sqrt();
}

Eigen::MatrixXd cov(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false) {
  throw_if(X.rows() < 2, "cov: need at least two observations");
  const auto mu = row_mean(X);
  Eigen::MatrixXd centred = X.rowwise() - mu;
  return (centred.adjoint() * centred) / static_cast<double>(X.rows() - 1);
}

Eigen::MatrixXd cov(const std::vector<Eigen::MatrixXd> &Xv) noexcept(false) {
  throw_if(Xv.empty(), "cov: vector is empty");

  const Index D = Xv.front().cols();
  Index Ntotal = 0;
  Eigen::RowVectorXd mu = Eigen::RowVectorXd::Zero(D);

  // 1st pass — global mean
  for (const auto &X : Xv) {
    throw_if(X.cols() != D, "cov: inconsistent dimensions in vector");
    mu += X.colwise().sum();
    Ntotal += X.rows();
  }
  mu /= static_cast<double>(Ntotal);

  // 2nd pass — covariance accumulation
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(D, D);
  for (const auto &X : Xv) {
    Eigen::MatrixXd centred = X.rowwise() - mu;
    S.noalias() += centred.adjoint() * centred;
  }
  return S / static_cast<double>(Ntotal - 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Distance & likelihood helpers
// ─────────────────────────────────────────────────────────────────────────────
Eigen::VectorXd mahaldist(Eigen::Ref<const Eigen::MatrixXd> X,
                          Eigen::Ref<const Eigen::RowVectorXd> mu,
                          Eigen::Ref<const Eigen::MatrixXd> A) noexcept(false) {
  throw_if(X.cols() != mu.cols() || X.cols() != A.cols(),
           "mahaldist: incompatible dimensionality");
  throw_if(A.rows() != A.cols(), "mahaldist: A must be square");

  Eigen::LDLT<Eigen::MatrixXd> Aldl(A);

  throw_if((Aldl.vectorD().array() <= 0).any(),
           "mahaldist: A is not positive definite");

  Eigen::MatrixXd centred = (X.rowwise() - mu).transpose();
  return ((centred.array() * (Aldl.solve(centred)).array()).colwise().sum())
      .transpose();
}

Eigen::VectorXd logsumexp(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false) {
  const Eigen::VectorXd mx = X.rowwise().maxCoeff();
  const Eigen::ArrayXd se = (X.colwise() - mx).array().exp().rowwise().sum();
  return mx + se.log().matrix();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Linear‑algebra helpers
// ─────────────────────────────────────────────────────────────────────────────
double eigpower(Eigen::Ref<const Eigen::MatrixXd> A,
                Eigen::VectorXd &eigvec) noexcept(false) {
  throw_if(A.rows() != A.cols(), "eigpower: A must be square");

  if (A.rows() == 1) {
    eigvec.resize(1);
    eigvec.setOnes();
    return A(0, 0);
  }

  Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(A.rows(), -1.0, 1.0);
  double eigval = v.norm();
  eigvec = v / eigval;

  Eigen::VectorXd prev = eigvec;
  double delta = std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < kMaxIter && delta > kEigConvThresh; ++iter) {
    v.noalias() = A * eigvec;
    eigval = v.norm();
    eigvec = v / eigval;
    delta = (eigvec - prev).norm();
    prev = eigvec;
  }
  return eigval;
}

double logdet(Eigen::Ref<const Eigen::MatrixXd> A) noexcept(false) {
  throw_if(A.rows() != A.cols(), "logdet: A must be square");

  Eigen::VectorXd d = A.ldlt().vectorD();
  throw_if((d.array() <= 0).any(), "logdet: A is not positive definite");

  return d.array().log().sum();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Special‑function helpers
// ─────────────────────────────────────────────────────────────────────────────
Eigen::MatrixXd mxdigamma(Eigen::Ref<const Eigen::MatrixXd> X) noexcept {
  return X.unaryExpr([](double v) { return digamma_scalar(v); });
}

Eigen::MatrixXd mxlgamma(Eigen::Ref<const Eigen::MatrixXd> X) noexcept {
  return X.unaryExpr([](double v) { return std::lgamma(v); });
}

} // namespace probutils
