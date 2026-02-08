/**
 * Mixture-model merge utilities
 * -----------------------------
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
 * @date 2026-01-12
 */

#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "distributions.hpp"

namespace vctk::merge {

struct MixtureComponent {
  int total_obs{0};
  int component_obs{0};
  double weight{0.0};
  Eigen::RowVectorXd mean;
  Eigen::MatrixXd cov;
};

struct ObservationModel {
  MixtureComponent component;
  double score{0.0};
};

[[nodiscard]] bool checkCovarianceEquivalent(const MixtureComponent &prior,
                                             const MixtureComponent &test,
                                             double alpha);

[[nodiscard]] bool checkMeanEquivalent(const MixtureComponent &prior,
                                       const MixtureComponent &test,
                                       double alpha);

[[nodiscard]] bool checkComponentEquivalent(const MixtureComponent &prior,
                                            const MixtureComponent &test,
                                            double alpha);

[[nodiscard]] std::vector<double>
getPriorWeights(const std::vector<MixtureComponent> &gmm);

[[nodiscard]] std::vector<MixtureComponent>
updateObs(std::vector<MixtureComponent> gmm, const std::vector<int> &num_obs);

[[nodiscard]] std::vector<MixtureComponent>
pruneMixtureModel(std::vector<MixtureComponent> gmm, std::size_t trunc_level);

[[nodiscard]] std::vector<MixtureComponent> mergeMixtureModel(
    Eigen::Ref<const Eigen::MatrixXd> data, Eigen::Ref<const Eigen::MatrixXd> qZ,
    const std::vector<MixtureComponent> &prior_model,
    const std::vector<distributions::GaussWish> &curr_model,
    const distributions::StickBreak &curr_weight, double alpha,
    std::size_t trunc_level);

[[nodiscard]] ObservationModel
getMixtureComponent(const std::vector<MixtureComponent> &gmm,
                    Eigen::Ref<const Eigen::VectorXd> observation);

} // namespace vctk::merge
