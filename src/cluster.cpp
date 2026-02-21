#include "vctk.hpp"

#include "comutils.hpp"
#include "probutils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#if defined(VCTK_HAS_OPENMP)
#include <omp.h>
#endif

namespace vctk {
namespace {

using comutils::ArrayXb;
using comutils::GreedOrder;
using Eigen::ArrayXd;
using Eigen::ArrayXi;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct InternalFailure {
  ErrorCode code;
  ErrorContext context;
};

void validate_opts(const VbemOptions &opts) {
  if (opts.converge <= 0.0)
    throw std::invalid_argument("converge must be > 0");
  if (opts.fenergy_delta < 0.0)
    throw std::invalid_argument("fenergy_delta must be >= 0");
  if (opts.zero_cutoff <= 0.0)
    throw std::invalid_argument("zero_cutoff must be > 0");
  if (opts.split_refine_iters < 1)
    throw std::invalid_argument("split_refine_iters must be >= 1");
  if (opts.n_threads < 0)
    throw std::invalid_argument("n_threads must be >= 0");
  if (opts.min_split_cluster_obs < 2)
    throw std::invalid_argument("min_split_cluster_obs must be >= 2");
  if (opts.min_split_partition_obs < 1)
    throw std::invalid_argument("min_split_partition_obs must be >= 1");
  if (opts.prune_zero_cutoff <= 0.0)
    throw std::invalid_argument("prune_zero_cutoff must be > 0");
  if (opts.min_split_improvement < 0.0)
    throw std::invalid_argument("min_split_improvement must be >= 0");
}

void validate_matrix_finite(Eigen::Ref<const MatrixXd> X, const char *name) {
  if (!X.allFinite())
    throw std::invalid_argument(
        fmt::format("{} contains NaN or Inf values", name));
}

[[nodiscard]] std::optional<InternalFailure>
validate_qz_status(Eigen::Ref<const MatrixXd> qZ, const char *algorithm_name,
                   const int outer_iter, const int vbem_iter,
                   const double free_energy) {
  if (!qZ.allFinite()) {
    return InternalFailure{
        ErrorCode::InvalidProbabilities,
        ErrorContext{algorithm_name, "qz_validation", outer_iter,
                     static_cast<int>(qZ.cols()), free_energy,
                     fmt::format("qZ contains NaN or Inf values (vbem_iter={})",
                                 vbem_iter)}};
  }
  if ((qZ.array() < -1.0e-12).any()) {
    return InternalFailure{
        ErrorCode::InvalidProbabilities,
        ErrorContext{
            algorithm_name, "qz_validation", outer_iter,
            static_cast<int>(qZ.cols()), free_energy,
            fmt::format("qZ contains negative probabilities (vbem_iter={})",
                        vbem_iter)}};
  }
  return std::nullopt;
}

[[nodiscard]] int resolve_omp_threads(const VbemOptions &opts) {
#if defined(VCTK_HAS_OPENMP)
  if (opts.deterministic)
    return 1;
  return opts.n_threads;
#else
  (void)opts;
  return 0;
#endif
}

void emit_error_context(const VbemOptions &opts, const ErrorContext &ctx) {
  if (!opts.error_callback)
    return;
  try {
    opts.error_callback(ctx);
  } catch (...) {
  }
}

[[noreturn]] void throw_runtime_with_context(const VbemOptions &opts,
                                             ErrorCode code,
                                             const ErrorContext &ctx) {
  emit_error_context(opts, ctx);
  ErrorContext c = ctx;
  c.message = fmt::format("[{}][{}][outer_iter={}][clusters={}] {}",
                          c.algorithm, c.phase, c.outer_iteration, c.clusters,
                          c.message);
  throw RuntimeError(code, std::move(c));
}

template <class C>
ArrayXd update_ss(Eigen::Ref<const MatrixXd> X, Eigen::Ref<const MatrixXd> qZ,
                  std::vector<C> &clusters, const VbemOptions &opts,
                  const int omp_threads) {
  const Index K = qZ.cols();
  const ArrayXd Nk = qZ.colwise().sum();

  ArrayXi k_full = ArrayXi::Zero(1);
  ArrayXi k_empty = ArrayXi::Zero(0);

  if (!opts.sparse && K > 1) {
    k_full = ArrayXi::LinSpaced(K, 0, K - 1);
  } else if (opts.sparse) {
    comutils::arrfind((Nk >= opts.zero_cutoff), k_full, k_empty);
  }

#if defined(VCTK_HAS_OPENMP)
  if (omp_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(omp_threads)
    for (Index i = 0; i < k_full.size(); ++i) {
      const Index k = k_full(i);
      clusters[k].addobs(qZ.col(k), X);
    }
  } else {
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < k_full.size(); ++i) {
      const Index k = k_full(i);
      clusters[k].addobs(qZ.col(k), X);
    }
  }
#else
  for (Index i = 0; i < k_full.size(); ++i) {
    const Index k = k_full(i);
    clusters[k].addobs(qZ.col(k), X);
  }
#endif

  return Nk;
}

template <class W, class C>
double vbexpectation(Eigen::Ref<const MatrixXd> X, const W &weights,
                     const std::vector<C> &clusters, MatrixXd &qZ,
                     const VbemOptions &opts, const int omp_threads) {
  const Index K = static_cast<Index>(clusters.size());
  const Index N = X.rows();

  ArrayXi k_full = ArrayXi::Zero(1);
  ArrayXi k_empty = ArrayXi::Zero(0);

  if (!opts.sparse && K > 1) {
    k_full = ArrayXi::LinSpaced(K, 0, K - 1);
  } else if (opts.sparse) {
    comutils::arrfind((weights.getNk() >= opts.zero_cutoff), k_full, k_empty);
  }

  const Index n_full = k_full.size();

  MatrixXd logqZ = MatrixXd::Zero(N, n_full);
  const ArrayXd e_logz = weights.Elogweight();

#if defined(VCTK_HAS_OPENMP)
  if (omp_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(omp_threads)
    for (Index i = 0; i < n_full; ++i) {
      const Index k = k_full(i);
      logqZ.col(i) = e_logz(k) + clusters[k].Eloglike(X).array();
    }
  } else {
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n_full; ++i) {
      const Index k = k_full(i);
      logqZ.col(i) = e_logz(k) + clusters[k].Eloglike(X).array();
    }
  }
#else
  for (Index i = 0; i < n_full; ++i) {
    const Index k = k_full(i);
    logqZ.col(i) = e_logz(k) + clusters[k].Eloglike(X).array();
  }
#endif

  const VectorXd log_norm = probutils::logsumexp(logqZ);

  qZ.setZero(N, K);
#if defined(VCTK_HAS_OPENMP)
  if (omp_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(omp_threads)
    for (Index i = 0; i < n_full; ++i) {
      const Index k = k_full(i);
      qZ.col(k) = (logqZ.col(i).array() - log_norm.array()).exp().matrix();
    }
  } else {
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n_full; ++i) {
      const Index k = k_full(i);
      qZ.col(k) = (logqZ.col(i).array() - log_norm.array()).exp().matrix();
    }
  }
#else
  for (Index i = 0; i < n_full; ++i) {
    const Index k = k_full(i);
    qZ.col(k) = (logqZ.col(i).array() - log_norm.array()).exp().matrix();
  }
#endif
  for (Index i = 0; i < k_empty.size(); ++i)
    qZ.col(k_empty(i)).setZero();

  return -log_norm.sum();
}

template <class W, class C>
double fenergy(const W &weights, const std::vector<C> &clusters,
               const double fxz) {
  double fw = weights.fenergy();
  double fc = 0.0;
  for (const auto &c : clusters)
    fc += c.fenergy();
  return fw + fc + fxz;
}

template <class W, class C>
std::optional<InternalFailure>
vbem_status(Eigen::Ref<const MatrixXd> X, MatrixXd &qZ, W &weights,
            std::vector<C> &clusters, const double clusterprior,
            const VbemOptions &opts, const int max_iters, const int omp_threads,
            const char *algorithm_name, const int outer_iter, double &out_f) {
  const Index K = qZ.cols();

  weights = W();
  clusters.assign(static_cast<std::size_t>(K), C(clusterprior, X.cols()));

  double f = std::numeric_limits<double>::infinity();
  int iter = 0;

  try {
    while (true) {
      const double f_prev = f;

      for (auto &c : clusters)
        c.clearobs();

      const ArrayXd nk = update_ss<C>(X, qZ, clusters, opts, omp_threads);
      weights.update(nk);

#if defined(VCTK_HAS_OPENMP)
      if (omp_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(omp_threads)
        for (Index k = 0; k < static_cast<Index>(clusters.size()); ++k) {
          clusters[static_cast<std::size_t>(k)].update();
        }
      } else {
#pragma omp parallel for schedule(static)
        for (Index k = 0; k < static_cast<Index>(clusters.size()); ++k) {
          clusters[static_cast<std::size_t>(k)].update();
        }
      }
#else
      for (Index k = 0; k < static_cast<Index>(clusters.size()); ++k) {
        clusters[static_cast<std::size_t>(k)].update();
      }
#endif

      const double fz =
          vbexpectation<W, C>(X, weights, clusters, qZ, opts, omp_threads);
      if (auto qz_failure = validate_qz_status(
              qZ, algorithm_name, outer_iter, iter, fz)) {
        return qz_failure;
      }
      f = fenergy<W, C>(weights, clusters, fz);

      if (std::isfinite(f_prev) && f_prev != 0.0) {
        const double rel_increase = (f - f_prev) / std::abs(f_prev);
        if (rel_increase > opts.fenergy_delta) {
          return InternalFailure{
              ErrorCode::FreeEnergyIncrease,
              ErrorContext{algorithm_name, "vbem", outer_iter,
                           static_cast<int>(clusters.size()), f,
                           fmt::format("Free energy increase (vbem_iter={})",
                                       iter)}};
        }

        const double rel_change = std::abs((f_prev - f) / f_prev);
        if (rel_change <= opts.converge)
          break;
      }

      ++iter;
      if (max_iters >= 0 && iter >= max_iters)
        break;
    }
  } catch (const std::exception &e) {
    return InternalFailure{
        ErrorCode::NumericalFailure,
        ErrorContext{algorithm_name, "vbem", outer_iter,
                     static_cast<int>(clusters.size()), f, e.what()}};
  } catch (...) {
    return InternalFailure{
        ErrorCode::NumericalFailure,
        ErrorContext{algorithm_name, "vbem", outer_iter,
                     static_cast<int>(clusters.size()), f,
                     "unknown vbem exception"}};
  }

  out_f = f;
  return std::nullopt;
}

template <class W, class C>
bool split_greedy(Eigen::Ref<const MatrixXd> X, const W &weights,
                  const std::vector<C> &clusters, MatrixXd &qZ,
                  std::vector<int> &tally, const double f,
                  const double clusterprior, const VbemOptions &opts,
                  const int omp_threads, const char *algorithm_name,
                  const int outer_iter, std::optional<InternalFailure> &failure) {
  failure.reset();
  const Index K = static_cast<Index>(clusters.size());
  try {
    if (opts.max_clusters >= 0 && K >= static_cast<Index>(opts.max_clusters))
      return false;

    tally.resize(static_cast<std::size_t>(K), 0);
    std::vector<GreedOrder> ord(static_cast<std::size_t>(K));

    for (Index k = 0; k < K; ++k) {
      ord[static_cast<std::size_t>(k)].k = static_cast<int>(k);
      ord[static_cast<std::size_t>(k)].tally =
          tally[static_cast<std::size_t>(k)];
      ord[static_cast<std::size_t>(k)].Fk =
          clusters[static_cast<std::size_t>(k)].fenergy();
    }

    const ArrayXd logpi = weights.Elogweight();
#if defined(VCTK_HAS_OPENMP)
    if (omp_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(omp_threads)
      for (Index k = 0; k < K; ++k) {
        const VectorXd qcol = qZ.col(k);
        VectorXd loglike = clusters[static_cast<std::size_t>(k)].Eloglike(X);
        loglike.array() += logpi(k);
        ord[static_cast<std::size_t>(k)].Fk -= qcol.dot(loglike);
      }
    } else {
#pragma omp parallel for schedule(static)
      for (Index k = 0; k < K; ++k) {
        const VectorXd qcol = qZ.col(k);
        VectorXd loglike = clusters[static_cast<std::size_t>(k)].Eloglike(X);
        loglike.array() += logpi(k);
        ord[static_cast<std::size_t>(k)].Fk -= qcol.dot(loglike);
      }
    }
#else
    for (Index k = 0; k < K; ++k) {
      const VectorXd qcol = qZ.col(k);
      VectorXd loglike = clusters[static_cast<std::size_t>(k)].Eloglike(X);
      loglike.array() += logpi(k);
      ord[static_cast<std::size_t>(k)].Fk -= qcol.dot(loglike);
    }
#endif

    std::stable_sort(ord.begin(), ord.end(), [](const GreedOrder &a,
                                                const GreedOrder &b) {
      if (a.tally != b.tally)
        return a.tally < b.tally;
      if (a.Fk != b.Fk)
        return a.Fk > b.Fk;
      return a.k < b.k;
    });

    for (const auto &candidate : ord) {
      const Index k = static_cast<Index>(candidate.k);
      ++tally[static_cast<std::size_t>(k)];

      if (clusters[static_cast<std::size_t>(k)].getN() <
          static_cast<double>(opts.min_split_cluster_obs))
        continue;

      MatrixXd Xk;
      const ArrayXi map_idx =
          comutils::partobs(X, (qZ.col(k).array() > 0.5), Xk);

      const Index m = Xk.rows();
      if (m < static_cast<Index>(2 * opts.min_split_partition_obs))
        continue;

      const ArrayXb splitk =
          clusters[static_cast<std::size_t>(k)].splitobs(Xk);
      const Index scount = splitk.count();
      if (scount < static_cast<Index>(opts.min_split_partition_obs) ||
          scount > (m - static_cast<Index>(opts.min_split_partition_obs)))
        continue;

      MatrixXd qZref = MatrixXd::Zero(m, 2);
      qZref.col(0) = (splitk == true).cast<double>();
      qZref.col(1) = (splitk == false).cast<double>();

      W wsplit;
      std::vector<C> csplit;
      double f_refine = std::numeric_limits<double>::infinity();
      if (auto vbem_failure = vbem_status<W, C>(
              Xk, qZref, wsplit, csplit, clusterprior, opts,
              opts.split_refine_iters, omp_threads, algorithm_name, outer_iter,
              f_refine)) {
        failure = InternalFailure{
            ErrorCode::SplitFailure,
            ErrorContext{algorithm_name, "split_prune", outer_iter,
                         static_cast<int>(clusters.size()), f,
                         fmt::format("split refine failed: {}",
                                     vbem_failure->context.message)}};
        return false;
      }
      (void)f_refine;

      if (comutils::anyempty(csplit))
        continue;

      MatrixXd qZaug =
          comutils::auglabels(k, map_idx, (qZref.col(1).array() > 0.5), qZ);

      double fsplit = std::numeric_limits<double>::infinity();
      if (auto vbem_failure =
              vbem_status<W, C>(X, qZaug, wsplit, csplit, clusterprior, opts,
                                /*max_iters=*/1, omp_threads, algorithm_name,
                                outer_iter, fsplit)) {
        failure = InternalFailure{
            ErrorCode::SplitFailure,
            ErrorContext{algorithm_name, "split_prune", outer_iter,
                         static_cast<int>(clusters.size()), f,
                         fmt::format("split accept check failed: {}",
                                     vbem_failure->context.message)}};
        return false;
      }

      if (comutils::anyempty(csplit))
        continue;

      if (fsplit < f &&
          std::abs((f - fsplit) / f) > opts.min_split_improvement) {
        qZ = std::move(qZaug);
        tally[static_cast<std::size_t>(k)] = 0;
        return true;
      }
    }
  } catch (const std::exception &e) {
    failure = InternalFailure{
        ErrorCode::SplitFailure,
        ErrorContext{algorithm_name, "split_prune", outer_iter,
                     static_cast<int>(clusters.size()), f,
                     fmt::format("split phase failed: {}", e.what())}};
    return false;
  } catch (...) {
    failure = InternalFailure{
        ErrorCode::SplitFailure,
        ErrorContext{algorithm_name, "split_prune", outer_iter,
                     static_cast<int>(clusters.size()), f,
                     "split phase failed: unknown exception"}};
    return false;
  }

  return false;
}

template <class W, class C>
std::optional<InternalFailure>
prune_clusters_status(MatrixXd &qZ, W &weights, std::vector<C> &clusters,
                      const VbemOptions &opts, const char *algorithm_name,
                      const int outer_iter, const double free_energy,
                      bool &out_pruned) {
  out_pruned = false;
  try {
    const Index K = static_cast<Index>(clusters.size());
    ArrayXd Nk(K);
    for (Index k = 0; k < K; ++k)
      Nk(k) = clusters[static_cast<std::size_t>(k)].getN();

    ArrayXi eidx;
    ArrayXi fidx;
    comutils::arrfind(Nk < opts.prune_zero_cutoff, eidx, fidx);
    if (eidx.size() == 0)
      return std::nullopt;

    for (Index i = eidx.size() - 1; i >= 0; --i)
      clusters.erase(clusters.begin() + eidx(i));

    MatrixXd qZnew = MatrixXd::Zero(qZ.rows(), fidx.size());
    for (Index i = 0; i < fidx.size(); ++i)
      qZnew.col(i) = qZ.col(fidx(i));

    weights.update(qZnew.colwise().sum());
    qZ = std::move(qZnew);
    out_pruned = true;
    return std::nullopt;
  } catch (const std::exception &e) {
    return InternalFailure{
        ErrorCode::PruneFailure,
        ErrorContext{algorithm_name, "split_prune", outer_iter,
                     static_cast<int>(clusters.size()), free_energy,
                     fmt::format("prune failed: {}", e.what())}};
  } catch (...) {
    return InternalFailure{
        ErrorCode::PruneFailure,
        ErrorContext{algorithm_name, "split_prune", outer_iter,
                     static_cast<int>(clusters.size()), free_energy,
                     "prune failed: unknown exception"}};
  }
}

template <class W, class C>
double learn_single(Eigen::Ref<const MatrixXd> X, MatrixXd &qZ, W &weights,
                    std::vector<C> &clusters, const double clusterprior,
                    const VbemOptions &opts, const char *algorithm_name) {
  validate_opts(opts);

  if (X.rows() == 0)
    throw std::invalid_argument("X must have at least one observation");
  if (X.cols() == 0)
    throw std::invalid_argument("X must have at least one feature");
  validate_matrix_finite(X, "X");
  if (clusterprior <= 0.0)
    throw std::invalid_argument("clusterprior must be > 0");
  const int omp_threads = resolve_omp_threads(opts);

  qZ = MatrixXd::Ones(X.rows(), 1);

  std::vector<int> tally;
  bool split = true;
  double f = std::numeric_limits<double>::infinity();
  int outer_iter = 0;

  while (split) {
    bool pruned = false;
    if (auto vbem_failure =
            vbem_status<W, C>(X, qZ, weights, clusters, clusterprior, opts,
                              opts.max_vbem_iters, omp_threads, algorithm_name,
                              outer_iter, f)) {
      throw_runtime_with_context(opts, vbem_failure->code, vbem_failure->context);
    }

    if (auto prune_failure =
            prune_clusters_status<W, C>(qZ, weights, clusters, opts,
                                        algorithm_name, outer_iter, f, pruned)) {
      throw_runtime_with_context(opts, prune_failure->code,
                                 prune_failure->context);
    }

    std::optional<InternalFailure> split_failure = std::nullopt;
    split = split_greedy<W, C>(X, weights, clusters, qZ, tally, f, clusterprior,
                               opts, omp_threads, algorithm_name, outer_iter,
                               split_failure);
    if (split_failure) {
      throw_runtime_with_context(opts, split_failure->code,
                                 split_failure->context);
    }

    if (opts.verbose) {
      spdlog::info("[{}] outer_iter={} clusters={} free_energy={} pruned={} split={}",
                   algorithm_name, outer_iter, clusters.size(), f, pruned, split);
    }

    if (opts.progress_callback) {
      try {
        opts.progress_callback(
            LearnProgress{outer_iter, static_cast<int>(clusters.size()), f, pruned,
                          split});
      } catch (const std::exception &e) {
        throw_runtime_with_context(
            opts, ErrorCode::CallbackFailure,
            ErrorContext{algorithm_name, "progress_callback", outer_iter,
                         static_cast<int>(clusters.size()), f, e.what()});
      } catch (...) {
        throw_runtime_with_context(
            opts, ErrorCode::CallbackFailure,
            ErrorContext{algorithm_name, "progress_callback", outer_iter,
                         static_cast<int>(clusters.size()), f,
                         "unknown callback exception"});
      }
    }
    ++outer_iter;
  }

  return f;
}

} // namespace

double learnVDP(Eigen::Ref<const MatrixXd> X, MatrixXd &qZ,
                distributions::StickBreak &weights,
                std::vector<distributions::GaussWish> &clusters,
                const double clusterprior, const VbemOptions &opts) {
  MatrixXd qZ_new;
  distributions::StickBreak weights_new;
  std::vector<distributions::GaussWish> clusters_new;
  const double f = learn_single<distributions::StickBreak, distributions::GaussWish>(
      X, qZ_new, weights_new, clusters_new, clusterprior, opts, "learnVDP");
  qZ = std::move(qZ_new);
  weights = std::move(weights_new);
  clusters = std::move(clusters_new);
  return f;
}

double learnBGMM(Eigen::Ref<const MatrixXd> X, MatrixXd &qZ,
                 distributions::Dirichlet &weights,
                 std::vector<distributions::GaussWish> &clusters,
                 const double clusterprior, const VbemOptions &opts) {
  MatrixXd qZ_new;
  distributions::Dirichlet weights_new;
  std::vector<distributions::GaussWish> clusters_new;
  const double f = learn_single<distributions::Dirichlet, distributions::GaussWish>(
      X, qZ_new, weights_new, clusters_new, clusterprior, opts, "learnBGMM");
  qZ = std::move(qZ_new);
  weights = std::move(weights_new);
  clusters = std::move(clusters_new);
  return f;
}

double learnDGMM(Eigen::Ref<const MatrixXd> X, MatrixXd &qZ,
                 distributions::Dirichlet &weights,
                 std::vector<distributions::NormGamma> &clusters,
                 const double clusterprior, const VbemOptions &opts) {
  MatrixXd qZ_new;
  distributions::Dirichlet weights_new;
  std::vector<distributions::NormGamma> clusters_new;
  const double f = learn_single<distributions::Dirichlet, distributions::NormGamma>(
      X, qZ_new, weights_new, clusters_new, clusterprior, opts, "learnDGMM");
  qZ = std::move(qZ_new);
  weights = std::move(weights_new);
  clusters = std::move(clusters_new);
  return f;
}

double learnBEMM(Eigen::Ref<const MatrixXd> X, MatrixXd &qZ,
                 distributions::Dirichlet &weights,
                 std::vector<distributions::ExpGamma> &clusters,
                 const double clusterprior, const VbemOptions &opts) {
  if ((X.array() < 0.0).any())
    throw std::invalid_argument("BEMM expects non-negative observations");

  MatrixXd qZ_new;
  distributions::Dirichlet weights_new;
  std::vector<distributions::ExpGamma> clusters_new;
  const double f = learn_single<distributions::Dirichlet, distributions::ExpGamma>(
      X, qZ_new, weights_new, clusters_new, clusterprior, opts, "learnBEMM");
  qZ = std::move(qZ_new);
  weights = std::move(weights_new);
  clusters = std::move(clusters_new);
  return f;
}

} // namespace vctk
