
#include "distributions.hpp"
#include "probutils.hpp"
#include "vctk.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <limits>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>

namespace distributions {
namespace {

using probutils::eigpower;
using probutils::digamma;
using probutils::logdet;
using probutils::mahaldist;
using probutils::mxdigamma;
using probutils::mxlgamma;

[[nodiscard]] Eigen::ArrayXd enumdims(Index D) {
  if (D > 1)
    return Eigen::ArrayXd::LinSpaced(D, 1, D);
  return Eigen::ArrayXd::Ones(1);
}

constexpr auto obsdesc = [](const std::pair<int, double> &a,
                            const std::pair<int, double> &b) noexcept {
  return a.second > b.second;
};

void throw_if_non_finite(double value, const char *name) {
  if (!std::isfinite(value))
    throw vctk::RuntimeError(
        vctk::ErrorCode::NumericalFailure,
        vctk::ErrorContext{"distributions", "non_finite_check", -1, 0, 0.0,
                           fmt::format("{} is non-finite", name)});
}

template <typename Derived>
void throw_if_non_finite_mat(const Eigen::MatrixBase<Derived> &m,
                             const char *name) {
  if (!m.allFinite())
    throw vctk::RuntimeError(
        vctk::ErrorCode::NumericalFailure,
        vctk::ErrorContext{"distributions", "non_finite_check", -1, 0, 0.0,
                           fmt::format("{} contains non-finite values", name)});
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
//  Stick‑Breaking weight distribution
// ═════════════════════════════════════════════════════════════════════════════
StickBreak::StickBreak() { prior_fcalc(); }

StickBreak::StickBreak(double concentration) {
  if (concentration <= 0.0)
    throw std::invalid_argument("StickBreak: concentration must be > 0");

  alpha1_p_ = concentration;
  alpha1_ = Eigen::ArrayXd::Constant(1, concentration);
  prior_fcalc();
}

void StickBreak::prior_fcalc() noexcept {
  F_p_ = std::lgamma(alpha1_p_) + std::lgamma(alpha2_p_) -
         std::lgamma(alpha1_p_ + alpha2_p_);
}

void StickBreak::update(const Eigen::ArrayXd &Nk) {
  const Index K = Nk.size();

  // resize & cache
  Nk_ = Nk;
  alpha1_.resize(K);
  alpha1_ = alpha1_p_ + Nk;
  alpha2_.resize(K);
  E_logv_.resize(K);
  E_lognv_.resize(K);
  E_logpi_.resize(K);
  ordvec_.assign(K, {-1, -1});

  for (Index k = 0; k < K; ++k) {
    ordvec_[k] = {static_cast<int>(k), Nk(k)};
  }
  std::ranges::sort(ordvec_, obsdesc);

  const double N = Nk.sum();
  double cumNk = 0.0;
  double cumE_lognv = 0.0;

  for (auto [idx, _] : ordvec_) {
    cumNk += Nk(idx);
    alpha2_(idx) = alpha2_p_ + (N - cumNk);

    const double psi_sum = digamma(alpha1_(idx) + alpha2_(idx));
    E_logv_(idx) = digamma(alpha1_(idx)) - psi_sum;
    E_lognv_(idx) = digamma(alpha2_(idx)) - psi_sum;

    E_logpi_(idx) = E_logv_(idx) + cumE_lognv;
    cumE_lognv += E_lognv_(idx);
  }
}

double StickBreak::fenergy() const noexcept {
  const Index K = alpha1_.size();
  return K * F_p_ +
         (mxlgamma(alpha1_ + alpha2_).array() - mxlgamma(alpha1_).array() -
          mxlgamma(alpha2_).array() + (alpha1_ - alpha1_p_) * E_logv_ +
          (alpha2_ - alpha2_p_) * E_lognv_)
             .sum();
}

// ── Generalised Dirichlet (truncated stick) ──────────────────────────────────
void GDirichlet::update(const Eigen::ArrayXd &Nk) {
  StickBreak::update(Nk);
  if (ordvec_.empty())
    throw std::invalid_argument("GDirichlet::update: Nk must be non-empty");

  const int smallest =
      std::ranges::min_element(ordvec_, {}, &std::pair<int, double>::second)
          ->first;

  E_logpi_(smallest) -= E_logv_(smallest);
  E_logv_(smallest) = 0.0;
  E_lognv_(smallest) = 0.0; // undefined; set 0 for numerical stability
}

double GDirichlet::fenergy() const noexcept {
  const Index K = ordvec_.size();
  double sum = 0.0;
  for (Index i = 0; i < K - 1; ++i) {
    const int k = ordvec_[i].first;
    sum += std::lgamma(alpha1_(k) + alpha2_(k)) - std::lgamma(alpha1_(k)) -
           std::lgamma(alpha2_(k)) + (alpha1_(k) - alpha1_p_) * E_logv_(k) +
           (alpha2_(k) - alpha2_p_) * E_lognv_(k);
  }
  return (K - 1) * F_p_ + sum;
}

// ── Dirichlet weight distribution ────────────────────────────────────────────
Dirichlet::Dirichlet() {}

Dirichlet::Dirichlet(double alpha) {
  if (alpha <= 0.0)
    throw std::invalid_argument("Dirichlet: alpha must be > 0");
  alpha_p_ = alpha;
  alpha_ = Eigen::ArrayXd::Constant(1, alpha);
}

void Dirichlet::update(const Eigen::ArrayXd &Nk) {
  const Index K = Nk.size();
  Nk_ = Nk;
  alpha_.resize(K);
  alpha_ = alpha_p_ + Nk;
  E_logpi_.resize(K);
  E_logpi_ = probutils::mxdigamma(alpha_).array() - digamma(alpha_.sum());
}

double Dirichlet::fenergy() const noexcept {
  const Index K = alpha_.size();
  return std::lgamma(alpha_.sum()) - (alpha_p_ - 1.0) * E_logpi_.sum() +
         ((alpha_ - 1.0) * E_logpi_ - probutils::mxlgamma(alpha_).array())
             .sum() -
         std::lgamma(K * alpha_p_) + K * std::lgamma(alpha_p_);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Gaussian‑Wishart cluster
// ═════════════════════════════════════════════════════════════════════════════
GaussWish::GaussWish(double clustwidth, std::size_t D)
    : ClusterDist(clustwidth, D), nu_p_(static_cast<double>(D)),
      beta_p_(BETAPRIOR), m_p_(Eigen::RowVectorXd::Zero(D)) {
  if (D == 0)
    throw std::invalid_argument("GaussWish: D must be > 0");
  if (clustwidth <= 0.0)
    throw std::invalid_argument("GaussWish: clustwidth must be > 0");

  iW_p_ = nu_p_ * prior_ * Eigen::MatrixXd::Identity(D, D);
  logdW_p_ = -logdet(iW_p_);

  F_p_ = probutils::mxlgamma((nu_p_ + 1.0 - enumdims(D)).matrix() / 2.0).sum();
  clearobs();
}

void GaussWish::addobs(Eigen::Ref<const Eigen::VectorXd> qZk,
                       Eigen::Ref<const Eigen::MatrixXd> X) {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("GaussWish::addobs: dimension mismatch");
  if (qZk.rows() != X.rows())
    throw std::invalid_argument("GaussWish::addobs: length mismatch");

  const Eigen::MatrixXd qX = (X.array().colwise() * qZk.array()).matrix();
  N_s_ += qZk.sum();
  x_s_ += qZk.transpose() * X;
  xx_s_.noalias() += qX.transpose() * X;
}

void GaussWish::update() {
  Eigen::RowVectorXd xk = Eigen::RowVectorXd::Zero(D_);
  if (N_s_ > 0) {
    xk = x_s_ / N_s_;
  }

  Eigen::MatrixXd Sk = xx_s_ - xk.transpose() * x_s_;
  Eigen::RowVectorXd diff = xk - m_p_;

  N_ = N_s_;
  nu_ = nu_p_ + N_;
  beta_ = beta_p_ + N_;
  m_ = (beta_p_ * m_p_ + x_s_) / beta_;
  iW_ = iW_p_ + Sk + (beta_p_ * N_ / beta_) * diff.transpose() * diff;
  logdW_ = -logdet(iW_);

  throw_if_non_finite(N_, "GaussWish::N_");
  throw_if_non_finite(nu_, "GaussWish::nu_");
  throw_if_non_finite(beta_, "GaussWish::beta_");
  throw_if_non_finite(logdW_, "GaussWish::logdW_");
  throw_if_non_finite_mat(m_, "GaussWish::m_");
  throw_if_non_finite_mat(iW_, "GaussWish::iW_");
}

void GaussWish::clearobs() {
  nu_ = nu_p_;
  beta_ = beta_p_;
  m_ = m_p_;
  iW_ = iW_p_;
  logdW_ = logdW_p_;

  N_s_ = 0.0;
  x_s_ = Eigen::RowVectorXd::Zero(D_);
  xx_s_ = Eigen::MatrixXd::Zero(D_, D_);
}

Eigen::VectorXd GaussWish::Eloglike(Eigen::Ref<const Eigen::MatrixXd> X) const {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("GaussWish::Eloglike: dimension mismatch");

  const double sumpsi =
      mxdigamma((nu_ + 1.0 - enumdims(D_)).matrix() / 2.0).sum();
  const double const_term =
      0.5 * (sumpsi + logdW_ - D_ * (1.0 / beta_ + std::log(std::numbers::pi)));

  Eigen::VectorXd quad;
  try {
    quad = mahaldist(X, m_, iW_);
  } catch (const std::invalid_argument &e) {
    throw vctk::RuntimeError(
        vctk::ErrorCode::NumericalFailure,
        vctk::ErrorContext{"distributions", "GaussWish::Eloglike", -1, 0, 0.0,
                           fmt::format("GaussWish::Eloglike: {}", e.what())});
  }
  return const_term - 0.5 * nu_ * quad.array();
}

ArrayXb GaussWish::splitobs(Eigen::Ref<const Eigen::MatrixXd> X) const {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("GaussWish::splitobs: dimension mismatch");

  Eigen::VectorXd eigvec;
  (void)eigpower(iW_, eigvec); // principal eigenvector
  return (((X.rowwise() - m_) * eigvec).array()) >= 0;
}

double GaussWish::fenergy() const noexcept {
  const Eigen::ArrayXd l = enumdims(D_);
  const double sumpsi = mxdigamma((nu_ + 1.0 - l).matrix() / 2.0).sum();

  return F_p_ +
         (D_ * (beta_p_ / beta_ - 1.0 - nu_ - std::log(beta_p_ / beta_)) +
          nu_ * ((iW_.ldlt().solve(iW_p_)).trace() +
                 beta_p_ * mahaldist(m_, m_p_, iW_)(0)) +
          nu_p_ * (logdW_p_ - logdW_) + N_ * sumpsi) *
             0.5 -
         mxlgamma((nu_ + 1.0 - l).matrix() / 2.0).sum();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Normal‑Gamma cluster
// ═════════════════════════════════════════════════════════════════════════════
NormGamma::NormGamma(double clustwidth, std::size_t D)
    : ClusterDist(clustwidth, D), nu_p_(NUPRIOR), beta_p_(BETAPRIOR),
      m_p_(Eigen::RowVectorXd::Zero(D)) {
  if (D == 0)
    throw std::invalid_argument("NormGamma: D must be > 0");
  if (clustwidth <= 0.0)
    throw std::invalid_argument("NormGamma: clustwidth must be > 0");

  L_p_ = nu_p_ * prior_ * Eigen::RowVectorXd::Ones(D);
  logL_p_ = L_p_.array().log().sum();
  clearobs();
}

void NormGamma::addobs(Eigen::Ref<const Eigen::VectorXd> qZk,
                       Eigen::Ref<const Eigen::MatrixXd> X) {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("NormGamma::addobs: dimension mismatch");
  if (qZk.rows() != X.rows())
    throw std::invalid_argument("NormGamma::addobs: length mismatch");

  N_s_ += qZk.sum();
  x_s_ += qZk.transpose() * X;
  xx_s_ +=
      (X.array().square().colwise() * qZk.array()).colwise().sum().matrix();
}

void NormGamma::update() {
  Eigen::RowVectorXd xk = Eigen::RowVectorXd::Zero(D_);
  Eigen::RowVectorXd Sk = Eigen::RowVectorXd::Zero(D_);
  if (N_s_ > 0) {
    xk = x_s_ / N_s_;
    Sk = (xx_s_.array() - x_s_.array().square() / N_s_).matrix();
  }

  N_ = N_s_;
  beta_ = beta_p_ + N_;
  nu_ = nu_p_ + 0.5 * N_;
  m_ = (beta_p_ * m_p_ + x_s_) / beta_;
  L_ = L_p_ + 0.5 * Sk +
       (beta_p_ * N_ / (2 * beta_)) * (xk - m_p_).array().square().matrix();

  if ((L_.array() <= 0).any())
    throw std::invalid_argument("NormGamma: variance collapsed to ≤ 0");

  logL_ = L_.array().log().sum();
  throw_if_non_finite(N_, "NormGamma::N_");
  throw_if_non_finite(nu_, "NormGamma::nu_");
  throw_if_non_finite(beta_, "NormGamma::beta_");
  throw_if_non_finite(logL_, "NormGamma::logL_");
  throw_if_non_finite_mat(m_, "NormGamma::m_");
  throw_if_non_finite_mat(L_, "NormGamma::L_");
}

void NormGamma::clearobs() {
  nu_ = nu_p_;
  beta_ = beta_p_;
  m_ = m_p_;
  L_ = L_p_;
  logL_ = logL_p_;

  N_s_ = 0.0;
  x_s_ = Eigen::RowVectorXd::Zero(D_);
  xx_s_ = Eigen::RowVectorXd::Zero(D_);
}

Eigen::VectorXd NormGamma::Eloglike(Eigen::Ref<const Eigen::MatrixXd> X) const {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("NormGamma::Eloglike: dimension mismatch");

  Eigen::VectorXd quad = ((X.rowwise() - m_).array().square().matrix() *
                          L_.cwiseInverse().transpose());

  const double log2pi = std::log(2.0 * std::numbers::pi);
  return 0.5 * (D_ * (digamma(nu_) - log2pi - 1.0 / beta_) - logL_ -
                nu_ * quad.array());
}

ArrayXb NormGamma::splitobs(Eigen::Ref<const Eigen::MatrixXd> X) const {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("NormGamma::splitobs: dimension mismatch");

  Index idx;
  L_.maxCoeff(&idx);
  return (X.col(idx).array() - m_(idx)) >= 0;
}

double NormGamma::fenergy() const noexcept {
  Eigen::VectorXd invL = L_.cwiseInverse().transpose();
  return D_ * (std::lgamma(nu_p_) - std::lgamma(nu_) + N_ * digamma(nu_) * 0.5 -
               nu_) +
         0.5 * D_ *
             (std::log(beta_) - std::log(beta_p_) - 1.0 + beta_p_ / beta_) +
         (beta_p_ * nu_ * 0.5) * (m_ - m_p_).array().square().matrix() * invL +
         nu_p_ * (logL_ - logL_p_) + nu_ * L_p_ * invL;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Exponential‑Gamma cluster
// ═════════════════════════════════════════════════════════════════════════════
ExpGamma::ExpGamma(double obsmag, std::size_t D)
    : ClusterDist(obsmag, D), a_p_(APRIOR), b_p_(obsmag) {
  if (D == 0)
    throw std::invalid_argument("ExpGamma: D must be > 0");
  if (obsmag <= 0.0)
    throw std::invalid_argument("ExpGamma: obsmag must be > 0");
  clearobs();
}

void ExpGamma::addobs(Eigen::Ref<const Eigen::VectorXd> qZk,
                      Eigen::Ref<const Eigen::MatrixXd> X) {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("ExpGamma::addobs: dimension mismatch");
  if (qZk.rows() != X.rows())
    throw std::invalid_argument("ExpGamma::addobs: length mismatch");

  N_s_ += qZk.sum();
  x_s_ += qZk.transpose() * X;
}

void ExpGamma::update() {
  N_ = N_s_;
  a_ = a_p_ + N_;
  ib_ = (b_p_ + x_s_.array()).inverse().matrix();
  logb_ = -ib_.array().log().sum();
  throw_if_non_finite(N_, "ExpGamma::N_");
  throw_if_non_finite(a_, "ExpGamma::a_");
  throw_if_non_finite(logb_, "ExpGamma::logb_");
  throw_if_non_finite_mat(ib_, "ExpGamma::ib_");
}

void ExpGamma::clearobs() {
  a_ = a_p_;
  ib_ = Eigen::RowVectorXd::Constant(D_, 1.0 / b_p_);
  logb_ = D_ * std::log(b_p_);

  N_s_ = 0.0;
  x_s_ = Eigen::RowVectorXd::Zero(D_);
}

Eigen::VectorXd ExpGamma::Eloglike(Eigen::Ref<const Eigen::MatrixXd> X) const {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("ExpGamma::Eloglike: dimension mismatch");

  return D_ * digamma(a_) - logb_ - (a_ * X * ib_.transpose()).array();
}

ArrayXb ExpGamma::splitobs(Eigen::Ref<const Eigen::MatrixXd> X) const {
  if (X.cols() != static_cast<Index>(D_))
    throw std::invalid_argument("ExpGamma::splitobs: dimension mismatch");

  Eigen::ArrayXd proj = X * (a_ * ib_).transpose();
  return proj > proj.mean();
}

double ExpGamma::fenergy() const noexcept {
  return D_ * ((a_ - a_p_) * digamma(a_) - a_ - a_p_ * std::log(b_p_) -
               std::lgamma(a_) + std::lgamma(a_p_)) +
         b_p_ * a_ * ib_.sum() + a_p_ * logb_;
}

} // namespace distributions
