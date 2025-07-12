/**
 * Various statistical helpers
 * ------------------------
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
 * @date 2025-07-12
 */
#pragma once

#include <vector>
#include <Eigen/Core>

namespace probutils {

using Index = Eigen::Index;

// ═════════════════════════════════════════════════════════════════════════════
//  1. statistics
// ═════════════════════════════════════════════════════════════════════════════

[[nodiscard]] Eigen::RowVectorXd
mean(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false);

[[nodiscard]] Eigen::RowVectorXd
mean(const std::vector<Eigen::MatrixXd> &X) noexcept(false);

[[nodiscard]] Eigen::RowVectorXd
stdev(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false);

[[nodiscard]] Eigen::MatrixXd
cov(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false);

[[nodiscard]] Eigen::MatrixXd
cov(const std::vector<Eigen::MatrixXd> &X) noexcept(false);

// ═════════════════════════════════════════════════════════════════════════════
// 2. Distance & log‑sum utilities
// ═════════════════════════════════════════════════════════════════════════════

[[nodiscard]] Eigen::VectorXd
mahaldist(Eigen::Ref<const Eigen::MatrixXd> X,
          Eigen::Ref<const Eigen::RowVectorXd> mu,
          Eigen::Ref<const Eigen::MatrixXd> A) noexcept(false);

[[nodiscard]] Eigen::VectorXd
logsumexp(Eigen::Ref<const Eigen::MatrixXd> X) noexcept(false);

// ═════════════════════════════════════════════════════════════════════════════
// 3. Linear‑algebra helpers
// ═════════════════════════════════════════════════════════════════════════════

/**
 *  @brief  Power‑iteration to obtain dominant eigen‑pair.
 *
 *  @param  A       square, symmetric (recommended) matrix.
 *  @param  eigvec  (output) dominant eigen‑vector.
 *  @return         dominant eigen‑value.
 *  @throws         std::invalid_argument on non‑square input.
 */
[[nodiscard]] double eigpower(Eigen::Ref<const Eigen::MatrixXd> A,
                              Eigen::VectorXd &eigvec) noexcept(false);

/**
 *  @brief  Natural logarithm of |det(A)| for a positive‑definite matrix.
 *  @throws std::invalid_argument if A is non‑square or not PSD.
 */
[[nodiscard]] double
logdet(Eigen::Ref<const Eigen::MatrixXd> A) noexcept(false);

// ═════════════════════════════════════════════════════════════════════════════
// 4. Element‑wise special functions (never throw)
// ═════════════════════════════════════════════════════════════════════════════

[[nodiscard]] Eigen::MatrixXd
mxdigamma(Eigen::Ref<const Eigen::MatrixXd> X) noexcept;

[[nodiscard]] Eigen::MatrixXd
mxlgamma(Eigen::Ref<const Eigen::MatrixXd> X) noexcept;

} // namespace probutils
