#include "vctk_c.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vctk.hpp"

namespace {

void write_error(char *out, const int cap, const char *msg) {
  if (out == nullptr || cap <= 0) {
    return;
  }
  const std::size_t n = std::min<std::size_t>(std::strlen(msg), cap - 1);
  std::memcpy(out, msg, n);
  out[n] = '\0';
}

void clear_error(char *out, const int cap) {
  if (out != nullptr && cap > 0) {
    out[0] = '\0';
  }
}

} // namespace

int vctk_learn_vdp_labels(const double *x_row_major, const int n, const int d,
                          const vctk_vbem_options_c *opts, int *labels_out,
                          const int labels_len, int *n_clusters_out,
                          double *free_energy_out, char *error_message_out,
                          const int error_message_capacity) {
  clear_error(error_message_out, error_message_capacity);

  if (x_row_major == nullptr || opts == nullptr || labels_out == nullptr ||
      n_clusters_out == nullptr || free_energy_out == nullptr) {
    write_error(error_message_out, error_message_capacity,
                "null pointer argument");
    return VCTK_STATUS_BAD_PARAMETER;
  }
  if (n <= 0 || d <= 0 || labels_len < n) {
    write_error(error_message_out, error_message_capacity,
                "invalid shape or output length");
    return VCTK_STATUS_BAD_PARAMETER;
  }

  try {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        X(x_row_major, n, d);

    vctk::VbemOptions cpp_opts;
    cpp_opts.max_vbem_iters = opts->max_vbem_iters;
    cpp_opts.split_refine_iters = opts->split_refine_iters;
    cpp_opts.max_clusters = opts->max_clusters;
    cpp_opts.n_threads = opts->n_threads;
    cpp_opts.min_split_cluster_obs =
        (opts->min_split_cluster_obs > 0) ? opts->min_split_cluster_obs : 4;
    cpp_opts.min_split_partition_obs =
        (opts->min_split_partition_obs > 0) ? opts->min_split_partition_obs : 2;
    cpp_opts.converge = opts->converge;
    cpp_opts.fenergy_delta = opts->fenergy_delta;
    cpp_opts.zero_cutoff = opts->zero_cutoff;
    cpp_opts.prune_zero_cutoff =
        (opts->prune_zero_cutoff > 0.0) ? opts->prune_zero_cutoff : 0.1;
    cpp_opts.min_split_improvement =
        (opts->min_split_improvement >= 0.0) ? opts->min_split_improvement
                                             : 1.0e-5;
    cpp_opts.deterministic = (opts->deterministic != 0);
    cpp_opts.sparse = (opts->sparse != 0);

    Eigen::MatrixXd qZ;
    distributions::StickBreak weights;
    std::vector<distributions::GaussWish> clusters;

    const double F = vctk::learnVDP(X, qZ, weights, clusters, vctk::PRIORVAL,
                                    cpp_opts);

    for (int i = 0; i < n; ++i) {
      Eigen::Index idx = 0;
      (void)qZ.row(i).maxCoeff(&idx);
      labels_out[i] = static_cast<int>(idx);
    }
    *n_clusters_out = static_cast<int>(clusters.size());
    *free_energy_out = F;
    return VCTK_STATUS_OK;
  } catch (const std::invalid_argument &e) {
    write_error(error_message_out, error_message_capacity, e.what());
    return VCTK_STATUS_INVALID_ARGUMENT;
  } catch (const vctk::RuntimeError &e) {
    write_error(error_message_out, error_message_capacity, e.what());
    return VCTK_STATUS_RUNTIME_ERROR;
  } catch (const std::runtime_error &e) {
    write_error(error_message_out, error_message_capacity, e.what());
    return VCTK_STATUS_RUNTIME_ERROR;
  } catch (const std::exception &e) {
    write_error(error_message_out, error_message_capacity, e.what());
    return VCTK_STATUS_UNKNOWN_ERROR;
  } catch (...) {
    write_error(error_message_out, error_message_capacity, "unknown exception");
    return VCTK_STATUS_UNKNOWN_ERROR;
  }
}
