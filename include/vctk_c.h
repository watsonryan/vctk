#ifndef VCTK_C_H
#define VCTK_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vctk_status_code {
  VCTK_STATUS_OK = 0,
  VCTK_STATUS_INVALID_ARGUMENT = 1,
  VCTK_STATUS_RUNTIME_ERROR = 2,
  VCTK_STATUS_UNKNOWN_ERROR = 3,
  VCTK_STATUS_BAD_PARAMETER = 4
} vctk_status_code;

typedef struct vctk_vbem_options_c {
  int max_vbem_iters;
  int split_refine_iters;
  int max_clusters;
  int n_threads;
  int min_split_cluster_obs;
  int min_split_partition_obs;
  double converge;
  double fenergy_delta;
  double zero_cutoff;
  double prune_zero_cutoff;
  double min_split_improvement;
  int deterministic;
  int sparse;
} vctk_vbem_options_c;

int vctk_learn_vdp_labels(
    const double *x_row_major, int n, int d, const vctk_vbem_options_c *opts,
    int *labels_out, int labels_len, int *n_clusters_out, double *free_energy_out,
    char *error_message_out, int error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif // VCTK_C_H
