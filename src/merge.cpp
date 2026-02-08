#include "merge.hpp"
#include "vctk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>

namespace vctk::merge {
namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

constexpr double kMinAlpha = 1.0e-12;
constexpr double kReg = 1.0e-9;

void validate_component(const MixtureComponent &c) {
  if (c.mean.size() == 0) {
    throw std::invalid_argument("mixture component mean must be non-empty");
  }
  if (c.cov.rows() != c.cov.cols() || c.cov.rows() != c.mean.size()) {
    throw std::invalid_argument("mixture component covariance has invalid shape");
  }
  if (!c.mean.allFinite() || !c.cov.allFinite()) {
    throw std::invalid_argument("mixture component contains NaN/Inf values");
  }
}

void validate_alpha(const double alpha) {
  if (!(alpha > 0.0 && alpha < 1.0)) {
    throw std::invalid_argument("alpha must satisfy 0 < alpha < 1");
  }
}

MatrixXd make_pd(MatrixXd cov) {
  cov = 0.5 * (cov + cov.transpose());
  const Index d = cov.rows();
  double jitter = kReg;
  for (int i = 0; i < 6; ++i) {
    Eigen::LLT<MatrixXd> llt(cov);
    if (llt.info() == Eigen::Success) {
      return cov;
    }
    cov.diagonal().array() += jitter;
    jitter *= 10.0;
  }
  throw vctk::RuntimeError(
      vctk::ErrorCode::NumericalFailure,
      vctk::ErrorContext{"mergeMixtureModel", "make_pd", -1, 0, 0.0,
                         "failed to regularize covariance to SPD"});
}

double stable_mahal_sq(const RowVectorXd &delta, const MatrixXd &cov) {
  Eigen::LDLT<MatrixXd> ldlt(cov);
  if (ldlt.info() != Eigen::Success ||
      (ldlt.vectorD().array() <= 0.0).any()) {
    const MatrixXd reg = make_pd(cov);
    ldlt.compute(reg);
  }
  return delta * ldlt.solve(delta.transpose());
}

double equivalence_threshold(const int d, const double alpha) {
  const double a = std::max(alpha, kMinAlpha);
  return std::max(1.0, static_cast<double>(d) * (-2.0 * std::log(a)));
}

MixtureComponent from_current_component(const distributions::GaussWish &cluster,
                                        const double weight,
                                        const int data_cardinality) {
  MixtureComponent c;
  c.total_obs = std::max(1, data_cardinality);
  c.component_obs = std::max(1, static_cast<int>(std::lround(cluster.getN())));
  c.weight = std::max(0.0, weight);
  c.mean = cluster.getmean();
  c.cov = make_pd(cluster.getcov());
  return c;
}

MixtureComponent combine_components(const MixtureComponent &prior,
                                    const MixtureComponent &curr,
                                    const int data_cardinality) {
  validate_component(prior);
  validate_component(curr);
  if (prior.mean.size() != curr.mean.size()) {
    throw std::invalid_argument("component dimension mismatch during merge");
  }

  const double prior_mass =
      std::max(1.0, static_cast<double>(prior.total_obs) * prior.weight);
  const double curr_mass = std::max(1.0, static_cast<double>(curr.component_obs));
  const double denom = prior_mass + curr_mass;

  MixtureComponent out;
  out.total_obs = std::max(1, prior.total_obs + data_cardinality);
  out.component_obs =
      std::max(1, prior.component_obs + curr.component_obs);
  out.weight = static_cast<double>(out.component_obs) /
               static_cast<double>(out.total_obs);

  out.mean = ((prior_mass * prior.mean) + (curr_mass * curr.mean)) / denom;

  const MatrixXd prior_second =
      prior.cov + prior.mean.transpose() * prior.mean;
  const MatrixXd curr_second = curr.cov + curr.mean.transpose() * curr.mean;
  MatrixXd cov = ((prior_mass * prior_second) + (curr_mass * curr_second)) /
                     denom -
                 out.mean.transpose() * out.mean;
  out.cov = make_pd(std::move(cov));

  return out;
}

void renormalize_weights(std::vector<MixtureComponent> &gmm) {
  if (gmm.empty()) {
    return;
  }

  double s = 0.0;
  for (const auto &c : gmm) {
    s += std::max(1, c.component_obs);
  }
  if (s <= 0.0) {
    const double w = 1.0 / static_cast<double>(gmm.size());
    for (auto &c : gmm) {
      c.weight = w;
    }
    return;
  }

  for (auto &c : gmm) {
    c.weight = static_cast<double>(std::max(1, c.component_obs)) / s;
  }
}

double match_distance(const MixtureComponent &prior,
                      const MixtureComponent &test) {
  const RowVectorXd d = prior.mean - test.mean;
  const MatrixXd avg_cov = make_pd(0.5 * (prior.cov + test.cov));
  return stable_mahal_sq(d, avg_cov);
}

} // namespace

bool checkCovarianceEquivalent(const MixtureComponent &prior,
                               const MixtureComponent &test,
                               const double alpha) {
  validate_component(prior);
  validate_component(test);
  validate_alpha(alpha);

  if (prior.mean.size() != test.mean.size()) {
    throw std::invalid_argument("dimension mismatch in covariance check");
  }

  const MatrixXd prior_cov = make_pd(prior.cov);
  const MatrixXd test_cov = make_pd(test.cov);
  const MatrixXd diff = prior_cov - test_cov;
  const double rel =
      diff.norm() / std::max(prior_cov.norm(), std::sqrt(kReg));
  const double tol = std::max(0.15, std::sqrt(-std::log(alpha)) * 0.35);
  return rel <= tol;
}

bool checkMeanEquivalent(const MixtureComponent &prior,
                         const MixtureComponent &test, const double alpha) {
  validate_component(prior);
  validate_component(test);
  validate_alpha(alpha);

  if (prior.mean.size() != test.mean.size()) {
    throw std::invalid_argument("dimension mismatch in mean check");
  }

  const RowVectorXd delta = prior.mean - test.mean;
  const MatrixXd avg_cov = make_pd(0.5 * (prior.cov + test.cov));
  const double t = stable_mahal_sq(delta, avg_cov);
  const double threshold = equivalence_threshold(prior.mean.size(), alpha);
  return t <= threshold;
}

bool checkComponentEquivalent(const MixtureComponent &prior,
                              const MixtureComponent &test,
                              const double alpha) {
  return checkCovarianceEquivalent(prior, test, alpha) &&
         checkMeanEquivalent(prior, test, alpha);
}

std::vector<double> getPriorWeights(const std::vector<MixtureComponent> &gmm) {
  std::vector<double> out;
  out.reserve(gmm.size());
  for (const auto &c : gmm) {
    out.push_back(c.weight);
  }
  return out;
}

std::vector<MixtureComponent> updateObs(std::vector<MixtureComponent> gmm,
                                        const std::vector<int> &num_obs) {
  if (gmm.size() != num_obs.size()) {
    throw std::invalid_argument("gmm and num_obs must have matching size");
  }

  int N = 0;
  for (const int n : num_obs) {
    if (n < 0) {
      throw std::invalid_argument("num_obs entries must be >= 0");
    }
    N += n;
  }

  for (std::size_t i = 0; i < gmm.size(); ++i) {
    gmm[i].total_obs += N;
    gmm[i].component_obs += num_obs[i];
    if (gmm[i].total_obs > 0) {
      gmm[i].weight =
          static_cast<double>(gmm[i].component_obs) / gmm[i].total_obs;
    }
  }
  return gmm;
}

std::vector<MixtureComponent> pruneMixtureModel(std::vector<MixtureComponent> gmm,
                                                const std::size_t trunc_level) {
  if (trunc_level == 0) {
    return {};
  }
  if (gmm.size() <= trunc_level) {
    renormalize_weights(gmm);
    return gmm;
  }

  std::sort(gmm.begin(), gmm.end(), [](const MixtureComponent &a,
                                       const MixtureComponent &b) {
    return a.component_obs > b.component_obs;
  });
  gmm.resize(trunc_level);
  renormalize_weights(gmm);
  return gmm;
}

std::vector<MixtureComponent> mergeMixtureModel(
    Eigen::Ref<const MatrixXd> data, Eigen::Ref<const MatrixXd> qZ,
    const std::vector<MixtureComponent> &prior_model,
    const std::vector<distributions::GaussWish> &curr_model,
    const distributions::StickBreak &curr_weight, const double alpha,
    const std::size_t trunc_level) {
  validate_alpha(alpha);
  if (data.rows() <= 0 || data.cols() <= 0) {
    throw std::invalid_argument("data must be non-empty");
  }
  if (qZ.rows() != data.rows()) {
    throw std::invalid_argument("qZ rows must equal data rows");
  }
  if (!data.allFinite() || !qZ.allFinite()) {
    throw std::invalid_argument("data/qZ contain NaN or Inf values");
  }

  std::vector<MixtureComponent> current;
  const Eigen::ArrayXd w = curr_weight.Elogweight().exp();
  const Index use_k =
      std::min<Index>(static_cast<Index>(curr_model.size()), qZ.cols());
  current.reserve(static_cast<std::size_t>(use_k));

  for (Index k = 0; k < use_k; ++k) {
    const auto &c = curr_model[static_cast<std::size_t>(k)];
    if (c.getN() < 2.0) {
      continue;
    }
    const double wk = (k < w.size()) ? w(k) : 0.0;
    current.push_back(
        from_current_component(c, wk, static_cast<int>(data.rows())));
  }

  if (prior_model.empty()) {
    auto out = pruneMixtureModel(std::move(current), trunc_level);
    renormalize_weights(out);
    return out;
  }

  std::vector<MixtureComponent> out;
  out.reserve(prior_model.size() + current.size());
  std::vector<bool> prior_used(prior_model.size(), false);
  std::vector<bool> curr_used(current.size(), false);

  for (std::size_t i = 0; i < prior_model.size(); ++i) {
    validate_component(prior_model[i]);
    double best_score = std::numeric_limits<double>::infinity();
    int best_j = -1;
    for (std::size_t j = 0; j < current.size(); ++j) {
      if (curr_used[j]) {
        continue;
      }
      if (!checkComponentEquivalent(prior_model[i], current[j], alpha)) {
        continue;
      }
      const double score = match_distance(prior_model[i], current[j]);
      if (score < best_score) {
        best_score = score;
        best_j = static_cast<int>(j);
      }
    }
    if (best_j >= 0) {
      const std::size_t jj = static_cast<std::size_t>(best_j);
      out.push_back(combine_components(prior_model[i], current[jj],
                                       static_cast<int>(data.rows())));
      prior_used[i] = true;
      curr_used[jj] = true;
    }
  }

  const int prior_total = std::max(1, prior_model.front().total_obs);
  for (std::size_t j = 0; j < current.size(); ++j) {
    if (curr_used[j]) {
      continue;
    }
    auto c = current[j];
    c.total_obs = prior_total + static_cast<int>(data.rows());
    c.weight = static_cast<double>(c.component_obs) / c.total_obs;
    out.push_back(std::move(c));
  }

  for (std::size_t i = 0; i < prior_model.size(); ++i) {
    if (prior_used[i]) {
      continue;
    }
    auto p = prior_model[i];
    p.total_obs += static_cast<int>(data.rows());
    p.weight = static_cast<double>(p.component_obs) / std::max(1, p.total_obs);
    out.push_back(std::move(p));
  }

  bool merged = true;
  while (merged && out.size() > 1) {
    merged = false;
    for (std::size_t i = 0; i + 1 < out.size() && !merged; ++i) {
      for (std::size_t j = i + 1; j < out.size(); ++j) {
        if (!checkComponentEquivalent(out[i], out[j], alpha)) {
          continue;
        }
        out[i] = combine_components(out[i], out[j], 0);
        out.erase(out.begin() + static_cast<std::ptrdiff_t>(j));
        merged = true;
        break;
      }
    }
  }

  out = pruneMixtureModel(std::move(out), trunc_level);
  renormalize_weights(out);
  return out;
}

ObservationModel getMixtureComponent(const std::vector<MixtureComponent> &gmm,
                                     Eigen::Ref<const VectorXd> observation) {
  if (gmm.empty()) {
    throw std::invalid_argument("gmm must be non-empty");
  }
  if (observation.size() == 0) {
    throw std::invalid_argument("observation must be non-empty");
  }

  int best_idx = -1;
  double best_score = -std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < gmm.size(); ++i) {
    const auto &c = gmm[i];
    validate_component(c);
    if (c.mean.size() != observation.size()) {
      throw std::invalid_argument("observation/component dimension mismatch");
    }
    const RowVectorXd delta = observation.transpose() - c.mean;
    const double md2 = stable_mahal_sq(delta, c.cov);
    const double score = std::log(std::max(c.weight, kReg)) - 0.5 * md2;

    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }

  if (best_idx < 0) {
    throw vctk::RuntimeError(
        vctk::ErrorCode::NumericalFailure,
        vctk::ErrorContext{"getMixtureComponent", "selection", -1, 0, 0.0,
                           "failed to select a mixture component"});
  }

  return ObservationModel{gmm[static_cast<std::size_t>(best_idx)], best_score};
}

} // namespace vctk::merge
