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
  indtrue = make_indices(expression, true);
  indfalse = make_indices(expression, false);
}

[[nodiscard]] ArrayXi partobs(const MatrixXd &X, const ArrayXb &Xpart,
                              MatrixXd &Xk) {
  if (Xpart.size() != X.rows())
    throw std::invalid_argument("partobs: Xpart size must equal X.rows()");

  const ArrayXi pidx = make_indices(Xpart, true);

  Xk.resize(pidx.size(), X.cols());
  for (Index row = 0; row < pidx.size(); ++row)
    Xk.row(row) = X.row(pidx(row));

  return pidx;
}

[[nodiscard]] MatrixXd auglabels(const Index k, const ArrayXi &map,
                                 const ArrayXb &Zsplit, const MatrixXd &qZ) {
  if (Zsplit.size() != map.size())
    throw std::invalid_argument("map and Zsplit must be the same size");

  const Index K = qZ.cols();
  if (K <= 0)
    throw std::invalid_argument("qZ must have at least one column");
  if (k < 0 || k >= K)
    throw std::out_of_range("k must index an existing qZ column");

  const ArrayXi split_idx = make_indices(Zsplit, true);

  MatrixXd qZaug = qZ;
  qZaug.conservativeResize(Eigen::NoChange, K + 1);
  qZaug.col(K).setZero();

  const Index k_col = k;
  for (Index i = 0; i < split_idx.size(); ++i) {
    const Index row = map(split_idx(i));
    if (row < 0 || row >= qZaug.rows())
      throw std::out_of_range("map contains row index outside qZ");
    qZaug(row, K) = qZaug(row, k_col);
    qZaug(row, k_col) = 0.0;
  }

  return qZaug;
}

} // namespace comutils
