/**
 * Main VCTK algorithm API
 * -----------------------
 *
 *                        _..._
 *                     .-'_..._''.
 *  .----.     .----..' .'      '.\          .
 *   \    \   /    // .'                   .'|
 *    '   '. /'   /. '               .|  .'  |
 *    |    |'    / | |             .' |_<    |
 *    |    ||    | | |           .'     ||   | ____
 *    '.   `'   .' . '          '--.  .-'|   | \ .'
 *     \        /   \ '.          .|  |  |   |/  .
 *      \      /     '. `._____.-'/|  |  |    /\  \
 *       '----'        `-.______ / |  '.'|   |  \  \
 *                              `  |   / '    \  \  \
 *                                 `'-' '------'  '---'
 *
 * @author watson
 * @date 2026-02-02
 */

#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "distributions.hpp"
#include "merge.hpp"

namespace vctk {

inline constexpr double PRIORVAL = 1.0;

struct LearnProgress {
  int outer_iteration{0};
  int clusters{0};
  double free_energy{0.0};
  bool pruned{false};
  bool split{false};
};

struct ErrorContext {
  std::string algorithm;
  std::string phase;
  int outer_iteration{0};
  int clusters{0};
  double free_energy{0.0};
  std::string message;
};

enum class ErrorCode {
  NumericalFailure,
  FreeEnergyIncrease,
  InvalidProbabilities,
  CallbackFailure,
  SplitFailure,
  PruneFailure
};

class RuntimeError : public std::runtime_error {
public:
  RuntimeError(ErrorCode code, ErrorContext context)
      : std::runtime_error(context.message), code_(code),
        context_(std::move(context)) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] const ErrorContext &context() const noexcept { return context_; }

private:
  ErrorCode code_;
  ErrorContext context_;
};

struct VbemOptions {
  int max_vbem_iters{-1};
  int split_refine_iters{15};
  int max_clusters{-1};
  int n_threads{0}; // 0 => runtime default
  std::uint64_t random_seed{0};
  int min_split_cluster_obs{4};
  int min_split_partition_obs{2};
  double converge{1.0e-5};
  double fenergy_delta{1.0e-6};
  double zero_cutoff{0.1};
  double prune_zero_cutoff{0.1};
  double min_split_improvement{1.0e-5};
  bool deterministic{false};
  bool sparse{false};
  bool verbose{false};
  // If set, called after each outer split/merge iteration.
  // If this callback throws, learning aborts and propagates as vctk::RuntimeError.
  std::function<void(const LearnProgress &)> progress_callback{};
  // Optional structured error telemetry callback.
  // Exceptions from this callback are swallowed.
  std::function<void(const ErrorContext &)> error_callback{};
};

// Exception contract:
// - std::invalid_argument: invalid shape/range/non-finite input parameters.
// - vctk::RuntimeError: numerical/optimization failures during fitting.
// Strong exception guarantee:
// - on exception, qZ/weights/clusters outputs are left unchanged.
[[nodiscard]] double learnVDP(
    Eigen::Ref<const Eigen::MatrixXd> X, Eigen::MatrixXd &qZ,
    distributions::StickBreak &weights,
    std::vector<distributions::GaussWish> &clusters,
    double clusterprior = PRIORVAL, const VbemOptions &opts = {});

[[nodiscard]] double learnBGMM(
    Eigen::Ref<const Eigen::MatrixXd> X, Eigen::MatrixXd &qZ,
    distributions::Dirichlet &weights,
    std::vector<distributions::GaussWish> &clusters,
    double clusterprior = PRIORVAL, const VbemOptions &opts = {});

[[nodiscard]] double learnDGMM(
    Eigen::Ref<const Eigen::MatrixXd> X, Eigen::MatrixXd &qZ,
    distributions::Dirichlet &weights,
    std::vector<distributions::NormGamma> &clusters,
    double clusterprior = PRIORVAL, const VbemOptions &opts = {});

[[nodiscard]] double learnBEMM(
    Eigen::Ref<const Eigen::MatrixXd> X, Eigen::MatrixXd &qZ,
    distributions::Dirichlet &weights,
    std::vector<distributions::ExpGamma> &clusters,
    double clusterprior = PRIORVAL, const VbemOptions &opts = {});

} // namespace vctk
