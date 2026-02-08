#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "vctk_c.h"

namespace {

void test_c_api_success() {
  const int n = 8;
  const int d = 2;
  const std::vector<double> X = {
      -2.0, -2.1, -1.9, -2.0, -2.2, -1.8, -1.7, -2.3,
      2.0,  2.1,  1.9,  2.2,  2.3,  1.8,  2.1,  1.7,
  };

  vctk_vbem_options_c opts{};
  opts.max_vbem_iters = 60;
  opts.split_refine_iters = 10;
  opts.max_clusters = 4;
  opts.n_threads = 1;
  opts.converge = 1.0e-5;
  opts.fenergy_delta = 1.0e-6;
  opts.zero_cutoff = 0.1;
  opts.deterministic = 1;
  opts.sparse = 0;

  std::vector<int> labels(n, -1);
  int n_clusters = 0;
  double f = 0.0;
  char err[256];

  const int rc = vctk_learn_vdp_labels(X.data(), n, d, &opts, labels.data(),
                                       static_cast<int>(labels.size()),
                                       &n_clusters, &f, err, sizeof(err));
  assert(rc == VCTK_STATUS_OK);
  assert(n_clusters > 0);
  assert(std::strlen(err) == 0);
}

void test_c_api_error_translation() {
  const int n = 2;
  const int d = 2;
  const std::vector<double> X = {
      0.0, 1.0, 2.0, 1.0 / 0.0,
  };

  vctk_vbem_options_c opts{};
  opts.max_vbem_iters = 40;
  opts.split_refine_iters = 8;
  opts.max_clusters = 4;
  opts.n_threads = 1;
  opts.converge = 1.0e-5;
  opts.fenergy_delta = 1.0e-6;
  opts.zero_cutoff = 0.1;
  opts.deterministic = 1;
  opts.sparse = 0;

  std::vector<int> labels(n, -1);
  int n_clusters = 0;
  double f = 0.0;
  char err[256];

  const int rc = vctk_learn_vdp_labels(X.data(), n, d, &opts, labels.data(),
                                       static_cast<int>(labels.size()),
                                       &n_clusters, &f, err, sizeof(err));
  assert(rc == VCTK_STATUS_INVALID_ARGUMENT);
  assert(std::strlen(err) > 0);
}

} // namespace

int main() {
  test_c_api_success();
  test_c_api_error_translation();
  std::cout << "All vctk C API tests passed\n";
  return 0;
}
