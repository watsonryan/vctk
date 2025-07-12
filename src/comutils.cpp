/**
 * variational‑cluster‑toolkit — modern rewrite of libcluster
 * License‑Identifier: LGPL‑3.0-or-later
 */
#include "comutils.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace comutils {

using Eigen::ArrayXi;
using Eigen::MatrixXd;
using Index = Eigen::Index;

[[nodiscard]] static ArrayXi make_indices(const ArrayXb &mask,
                                          const bool want_true) {
  const Index N = mask.size();
  const Index M = want_true ? mask.count() : N - mask.count();

  ArrayXi idx(M);
  Index cursor = 0;

  for (Index i = 0; i < N; ++i)
    if (mask(i) == want_true)
      idx(cursor++) = i;

  return idx;
}

void arrfind(const ArrayXb &expression, ArrayXi &indtrue, ArrayXi &indfalse) {
  indtrue = make_indices(expression, /*want_true=*/true);
  indfalse = make_indices(expression, /*want_true=*/false);
}

[[nodiscard]] ArrayXi partobs(const MatrixXd &X, const ArrayXb &Xpart,
                              MatrixXd &Xk) {
  const ArrayXi pidx = make_indices(Xpart, /*want_true=*/true);

  Xk.resize(pidx.size(), X.cols());
  for (Index row = 0; row < pidx.size(); ++row)
    Xk.row(row) = X.row(pidx(row));

  return pidx; // copy‑elided (NRVO / move)
}

// ---------------------------------------------------------------------------

[[nodiscard]] MatrixXd auglabels(const double k, const ArrayXi &map,
                                 const ArrayXb &Zsplit, const MatrixXd &qZ) {
  if (Zsplit.size() != map.size())
    throw std::invalid_argument("map and Zsplit must be the same size");

  const Index K = qZ.cols();
  const ArrayXi split_idx = make_indices(Zsplit, /*want_true=*/true);

  MatrixXd qZaug = qZ; // deep copy
  qZaug.conservativeResize(Eigen::NoChange, K + 1);
  qZaug.col(K).setZero();

  const Index k_col = static_cast<Index>(k);
  for (Index i = 0; i < split_idx.size(); ++i) {
    const Index row = map(split_idx(i));
    qZaug(row, K) = qZaug(row, k_col);
    qZaug(row, k_col) = 0.0;
  }

  return qZaug;
}

} // namespace comutils
