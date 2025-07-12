#pragma once // Modern, non‑intrusive include guard

#include <Eigen/Dense>
#include <algorithm>
#include <concepts>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace comutils {

typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;

/**
 * @brief  Information required for greedy cluster‑split ordering.
 *
 * Ordering rules (see @ref operator<):
 * 1. Prefer clusters with *fewer* previous failed splits (`tally`).
 * 2. If tied, prefer clusters that contribute *more* free energy (`Fk`).
 */
struct GreedOrder {
  int k{0};       ///< Cluster index
  int tally{0};   ///< # times this cluster failed to split
  double Fk{0.0}; ///< Approximate free‑energy contribution

  [[nodiscard]] constexpr bool
  operator<(const GreedOrder &other) const noexcept {
    return (tally == other.tally) ? (Fk > other.Fk) : (tally < other.tally);
  }
};

/* Backward‑compatibility alias for existing call‑sites */
[[nodiscard]] inline constexpr bool greedcomp(const GreedOrder &lhs,
                                              const GreedOrder &rhs) noexcept {
  return lhs < rhs;
}

/** Locate indices of `true` and `false` entries in the order encountered. */
void arrfind(const ArrayXb &expression, Eigen::ArrayXi &indtrue,
             Eigen::ArrayXi &indfalse);

/** Partition observations `X` by logical mask `Xpart`.
 *
 * @return  *M × 1* array mapping the rows in `Xk` back to their positions in
 *          the original `X`.
 */
[[nodiscard]] Eigen::ArrayXi partobs(const Eigen::MatrixXd &X,
                                     const ArrayXb &Xpart,
                                     Eigen::MatrixXd &Xk);

/** Augment assignment matrix `qZ` with a new split column.
 *
 * @throws std::invalid_argument  if `map.size() != Zsplit.size()`.
 * @return *(N × (K + 1))* assignment probability matrix.
 */
[[nodiscard]] Eigen::MatrixXd auglabels(double k, const Eigen::ArrayXi &map,
                                        const ArrayXb &Zsplit,
                                        const Eigen::MatrixXd &qZ);

/**
 * @brief Checks if any cluster in the given range is empty or contains only one
 * element.
 *
 * This function iterates over the provided range of clusters and returns true
 * if at least one cluster has a size less than or equal to 1, as determined by
 * the `getN()` member function of the cluster.
 *
 * @tparam Range A type that models `std::ranges::input_range` whose value type
 * provides a `getN()` method returning a value convertible to `std::size_t`.
 * @param clusters The range of cluster objects to check.
 * @return true if any cluster in the range is empty or contains only one
 * element; false otherwise.
 */
template <std::ranges::input_range Range>
requires requires(const std::ranges::range_value_t<Range> &c) {
  { c.getN() } -> std::convertible_to<std::size_t>;
}
[[nodiscard]] constexpr bool anyempty(const Range &clusters) noexcept {
  return std::ranges::any_of(
      clusters, [](const auto &c) noexcept { return c.getN() <= 1; });
}

} // namespace comutils